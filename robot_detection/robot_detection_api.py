from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import time
import os
import copy
import numpy as np
import cv2
import json
from scipy import ndimage
from PaddleOCR.predict_system import TextSystem, util, sorted_boxes, predict_det, predict_rec
from tools import func_timer, cv2ImgAddText, set_logger
import math
from AIConfig import *
import matplotlib.cm as cm


class Load_Model:
    def __init__(self):
        # 通过配置文件加载SOLO模型
        self.label = None
        self.mask = None
        self.img_show = None
        self.detect_results = []  # 对检测结果进行排序形成新的结果进行遍历
        self.whole_save_dir = test['save_dir'] + 'whole/'  # 大图保存路径
        if not os.path.exists(self.whole_save_dir):
            os.makedirs(self.whole_save_dir)
        self.logger = set_logger('test')  # 日志初始化
        # self.logger.info('··· is error')  # 用例

        self.model = init_detector(AI_weight['config_file'], AI_weight['checkpoint_file'], device='cuda:0')
        # 通过配置文件加载PaddleOCR模型
        ocr_args = util.parse_args()
        self.text_sys = TextSystem(ocr_args)

    # 得到裁切小图和对应的相关信息
    def get_crop_information(self, img):
        contours = self.mask2contour()  # 将mask转为边缘轮廓点
        for i, contour in enumerate(contours):
            # 外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            img_crop = img[y:y + h, x:x + w]
            self.detect_results.append([self.label, [x, y, w, h], self.mask, img_crop, contour])
        self.detect_results = sorted(self.detect_results, key=lambda a: a[1][1])  # 按纵坐标进行排序

    # 将mask转为边缘轮廓点
    def mask2contour(self):
        # #####分割图像的二值化处理######
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        ret, binary = cv2.threshold(self.mask, 0, 255, cv2.THRESH_BINARY)
        th2 = cv2.dilate(binary, kernel)

        # #########边缘轮廓点#######
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    # SOLO检测结果
    def SOLO_detect_result(self, SOLO_model, img, name, score_thr=0.3, sort_by_density=False):
        assert isinstance(SOLO_model.CLASSES, (tuple, list))

        # SOLO检测结果
        result = inference_detector(SOLO_model, img)
        img_show = img.copy()  # 原图
        h, w, _ = img.shape

        if not result or result == [None]:
            return img_show
        cur_result = result[0]
        seg_label = cur_result[0]
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()  # 类别
        score = cur_result[2].cpu().numpy()  # 置信度

        vis_inds = score > score_thr
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        if sort_by_density:
            mask_density = []
            for idx in range(num_mask):
                cur_mask = seg_label[idx, :, :]
                cur_mask = mmcv.imresize(cur_mask, (w, h))
                cur_mask = (cur_mask > 0.5).astype(np.int32)
                mask_density.append(cur_mask.sum())
            orders = np.argsort(mask_density)
            seg_label = seg_label[orders]
            cate_label = cate_label[orders]
            cate_score = cate_score[orders]

        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(num_mask)
        ]

        i = 0
        for idx in range(num_mask):
            idx = -(idx + 1)

            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]
            self.label = self.model.CLASSES[cur_cate]

            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            color_mask = color_masks[idx]
            cur_mask_bool = cur_mask.astype(np.bool)
            img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

            # 将上述分割图变成黑白二值图
            mask = cur_mask.copy()
            self.mask = mask * 255

            # 对检测结果进行排序处理
            self.get_crop_information(img)

            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))

            # 识别结果显示中文类别
            text = CN_names_param[self.label]
            img_show = cv2ImgAddText(img_show, text, vis_pos[0], vis_pos[1])

            # 单个实例的可视化
            per_img_show = img.copy()
            per_img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5
            per_img_show = cv2ImgAddText(per_img_show, text, vis_pos[0], vis_pos[1])
            img_save_dir = self.whole_save_dir + str(name)
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            cv2.imwrite(os.path.join(img_save_dir, '{}_{}_{}.jpg'.format(str(name), str(self.label), str(i))),
                        per_img_show)
            i += 1
        self.img_show = img_show


class Robot_Detection(Load_Model):
    # 初始化参数
    def __init__(self, json_info):
        # 保存路径
        super(Robot_Detection, self).__init__()
        self.save_dir = test['save_dir']
        self.crop_save_dir = test['save_dir'] + 'crop/'  # 小图保存路径
        # self.whole_save_dir = test['save_dir'] + 'whole/'  # 大图保存路径
        self.json_save_dir = test['save_dir'] + 'json/'  # json保存路径
        self.log_path = test['log_path']  # 日志保存路径
        if not os.path.exists(self.crop_save_dir):
            os.makedirs(self.crop_save_dir)
        # if not os.path.exists(self.whole_save_dir):
        #     os.makedirs(self.whole_save_dir)
        if not os.path.exists(self.json_save_dir):
            os.makedirs(self.json_save_dir)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.SOLO_model = self.model  # SOLO模型
        self.paddle_model = self.text_sys  # PaddleOCR模型
        self.json_info = json_info  # 默认json文件

        self.img_show = []  # 原图检测显示
        # self.detect_results = []  # 对检测结果进行排序形成新的结果进行遍历
        self.label = ''  # 标签
        self.bbox = []  # 检测矩形框
        self.mask = []  # 分割图
        self.crop = []  # 裁剪小图
        self.contour = []  # 轮廓信息
        self.now_crop_save_dir = ''  # 每个类别保存路径
        self.result = ''  # 检测目标的状态结果

    # 各个部件排序
    def object_sort(self, count_variable):
        tmp = {}  # 检测结果按类别存入字典
        for label in count_variable.keys():
            tmp[label] = [item for item in self.detect_results if item[0] == label]
        tmp['x'] = sorted(tmp['x'], key=lambda a: a[1][0])  # 旋钮部件排序
        tmp['y'] = sorted(tmp['y'], key=lambda a: a[1][0])  # 仪表部件排序
        i, end_list, slice_list = 0, [0], []
        while i < len(tmp['a']) - 1:
            y = tmp['a'][i][1][1]
            if y + 100 < tmp['a'][i + 1][1][1]:  # 两个按钮纵坐标之间的距离超过100
                end_list.append(i + 1)
                start = end_list.pop(0)
                slice_list.append(sorted(tmp['a'][start: i + 1], key=lambda a: a[1][0]))
            i += 1
        start = end_list.pop(0)
        slice_list.append(sorted(tmp['a'][start:], key=lambda a: a[1][0]))
        tmp['a'] = sum(slice_list, [])
        return tmp

    # 结果写入json
    def write_json(self, count, x, y, w, h, index):
        if '{}_{}'.format(EN_name_param[self.label], count) in self.json_info[index]["content"].keys():
            content = self.json_info[index]["content"]['{}_{}'.format(EN_name_param[self.label], count)]
            content["name"] = CN_names_param[self.label]
            content["result"] = self.result
            content["x"] = x
            content["y"] = y
            content["w"] = w
            content["h"] = h
        else:
            pass

    # 消防设施
    def fire_equipment_detect(self, name, count, index, count_variable):
        # 外接矩形
        x, y, w, h = self.bbox
        mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))
        self.result = "正常"
        print("图像中检测到 {} —— {}, 其状态为 {}" .format(self.label, CN_names_param[self.label], self.result))
        self.write_json(count_variable[self.label], x, y, w, h, index)
        count_variable[self.label] += 1

    # 通过按钮颜色识别按钮开关状态
    def judge_button_color(self, name, count, index, count_variable):
        # 外接矩形
        x, y, w, h = self.bbox
        if w > Threshold_parameter['button']['w'] or h > Threshold_parameter['button']['h']:
            mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))
            img_crop = cv2.resize(self.crop, (100, 100))
            crop_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            height = crop_gray.shape[0]
            width = crop_gray.shape[1]

            image_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
            H, S, V = image_hsv[height // 2, width // 2, :]
            self.result = "开" if V > Threshold_parameter['button']['V'] else "关"
            print("图像中检测到 {} —— {}, 其状态为 {}" .format(self.label, CN_names_param[self.label], self.result))
            self.write_json(count_variable[self.label], x, y, w, h, index)
            count_variable[self.label] += 1

    # 文字识别
    def recognize_OCR(self, name, count, index, count_variable):
        # 外接矩形
        x, y, w, h = self.bbox
        mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))
        if self.crop is None:
            print('图片为空！')

        # 文字识别
        dt_boxes, rec_res = self.paddle_model(self.crop)

        texts = []
        if self.label == 'y1':
            for text, score in rec_res:
                if text.isdigit() or text.split('.')[0].isdigit() or text[-1].isdigit():
                    text = text.replace(text[0], "6", 1)
                    if '.' not in text:
                        tmp = list(text)
                        tmp.insert(1, '.')
                        text = ''.join(tmp)
                    elif '..' in text:
                        text = text.replace("..", ".", 1)
                    else:
                        pass
                    texts.append(text)
                else:
                    pass
        else:
            for text, score in rec_res:
                texts.append(text)
        self.result = str(texts)
        print("图像中检测到 {} —— {}, 其状态为 {}".format(self.label, CN_names_param[self.label], self.result))
        self.write_json(count_variable[self.label], x, y, w, h, index)
        count_variable[self.label] += 1

    # 竖向阀门开关状态
    def shu_famen_recognize(self, name, count, index, count_variable):
        # 外接矩形
        x, y, w, h = self.bbox
        mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))
        if index == '53_4':
            self.result = "关闭"
        if index == '56_4':
            self.result = '开启'
        print("图像中检测到 {} —— {}, 其状态为 {}".format(self.label, CN_names_param[self.label], self.result))
        self.write_json(count_variable[self.label], x, y, w, h, index)
        count_variable[self.label] += 1

    # 横向阀门开关状态
    def heng_famen_recognize(self, name, count, index, count_variable):
        def mask2boxes(mask):
            boxes = []
            kernel = np.ones((3, 3), np.uint8)

            mask1 = cv2.dilate(mask, kernel, iterations=1)
            n_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask1, connectivity=4)
            for component_id in range(1, n_components):
                size = stats[component_id, cv2.CC_STAT_AREA]
                if size < 20:
                    continue
                left = stats[component_id, cv2.CC_STAT_LEFT]
                top = stats[component_id, cv2.CC_STAT_TOP]
                width = stats[component_id, cv2.CC_STAT_WIDTH]
                height = stats[component_id, cv2.CC_STAT_HEIGHT]
                boxes.append([left, top, left + width, top + height])
            return boxes

        # ###将小块分成3份，每份的面积
        def ComputEachArea(segmap, Rect, num):
            inter = int((Rect[3] - Rect[1]) / num)
            sum_area = 0
            each_area = []
            # 把小块分为num份，分别计算面积

            for i in range(num):
                count = 0
                left_point_x = inter * i + Rect[1]  # ##上面的y值
                right_point_x = inter * i + Rect[1] + inter  # ##下面面的y值
                if right_point_x > Rect[3]:
                    right_point_x = Rect[3]
                height = right_point_x - left_point_x  # ##小块的高
                for w1 in range(Rect[0], Rect[2]):
                    for h1 in range(left_point_x, right_point_x):
                        if segmap[h1, w1] > 0:
                            # sum_y = sum_y+(HEIGHT[1] - h)
                            count = count + 1
                each_area.append(count)

            return each_area

        def judge_famen_open(img1):
            num = 3
            open_or_not = "关闭"
            box = mask2boxes(img1)
            areas = ComputEachArea(img1, box[0], num)
            if areas[0] / max(areas) < 0.2:
                open_or_not = "开启"
            return open_or_not

        # 外接矩形
        x, y, w, h = self.bbox
        mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))

        open_or_not = judge_famen_open(self.mask)
        self.result = str(open_or_not)
        print("图像中检测到 {} —— {}, 其状态为 {}".format(self.label, CN_names_param[self.label], self.result))
        self.write_json(count_variable[self.label], x, y, w, h, index)
        count_variable[self.label] += 1

    # 水表识别
    def recognize_round_meter(self, name, count, index, count_variable):
        def th_2value(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            HH, WW = gray.shape
            interH = int(HH / 5)
            interW = int(WW / 5)
            size_threshold = HH * WW // 100 - 5

            crop_gray = gray[interH: HH - interH, interW: WW - interW]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            ret, thresh = cv2.threshold(crop_gray, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh = cv2.dilate(thresh, kernel)

            n_components, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)
            for component_id in range(1, n_components):
                # Filter by size
                size = stats[component_id, cv2.CC_STAT_AREA]
                if size < size_threshold:
                    continue
                else:
                    segmap = np.zeros_like(thresh)
                    segmap[labels == component_id] = 255
                    return segmap, crop_gray
            return None, None

        # 外接矩形
        x, y, w, h = self.bbox
        mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))

        if index == '13_4' or index == '23_4' or index == '32_4':
            crop = cv2.flip(self.crop, -1)
        elif index == '21_3' or index == '31_3':
            temp = cv2.transpose(self.crop)
            crop = cv2.flip(temp, 0)
        else:
            crop = self.crop

        crop = cv2.resize(crop, (100, 100))
        segment, gray_crop = th_2value(crop)
        if segment is not None and gray_crop is not None:
            # 二值化轮廓
            image_line = np.zeros_like(gray_crop)
            gray_color = cv2.cvtColor(image_line, cv2.COLOR_GRAY2BGR)
            contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 画出二值化轮廓
            crop = cv2.drawContours(gray_color, contours, -1, (255, 255, 255), 1)  # img为三通道才能显示轮廓
            img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            thresholds = Threshold_parameter['round_meter']['line']
            for threshold in thresholds:
                lines = cv2.HoughLines(img_gray, 1, np.pi / 180, threshold)
                if lines is None:
                    pass
                elif lines[0][0][1] == 0 and len(lines) == 1:
                    pass
                else:
                    thetas = []
                    for line in lines:
                        rho, theta = line[0]
                        if 0 < theta < 3:
                            thetas.append(theta)
                    if len(thetas) > 0:
                        average_theta = sum(thetas) / len(thetas)
                        if index == '12_3' or index == '22_4' or index == '32_3':
                            reading = (average_theta - meter_start[index][str(count_variable[self.label])]
                                       * math.pi / 180) / (math.pi * 1.5) * meter_range[index][
                                          str(count_variable[self.label])]
                        else:
                            reading = (average_theta - meter_start[index] * math.pi / 180) / (
                                    math.pi * 1.5) * meter_range[index]
                        self.result = abs(reading)
                        self.result = str(round(self.result, 2))
                    break
        print("图像中检测到 {} —— {}, 其读数为 {}".format(self.label, CN_names_param[self.label], self.result))
        self.write_json(count_variable[self.label], x, y, w, h, index)
        count_variable[self.label] += 1

    # 仪表识别
    def recognize_square_meter(self, name, count, index, count_variable):
        def th_2value(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            HH, WW = gray.shape

            interH = int(HH / 5)
            interW = int(WW / 5)
            size_threshold = HH * WW / 100
            gray_crop = gray[interH: HH - interH, interW: WW - interW]

            ret, thresh = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            n_components, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)

            for component_id in range(1, n_components):
                # Filter by size
                size = stats[component_id, cv2.CC_STAT_AREA]

                if size < size_threshold:
                    continue
                segmap = np.zeros_like(thresh)
                segmap[labels == component_id] = 255
                return segmap, gray_crop

        # 外接矩形
        x, y, w, h = self.bbox
        mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))

        HH, WW, _ = self.crop.shape
        if HH / WW > 1.3 or HH / WW < 0.8:
            pass
        else:
            img = cv2.resize(self.crop, (100, 100))
            segment, gray_crop = th_2value(img)
            # 二值化轮廓
            image_line = np.zeros_like(gray_crop)
            gray_color = cv2.cvtColor(image_line, cv2.COLOR_GRAY2BGR)
            contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 画出二值化轮廓
            img = cv2.drawContours(gray_color, contours, -1, (255, 255, 255), 1)  # img为三通道才能显示轮廓
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresholds = Threshold_parameter['square_meter']['line']
            for threshold in thresholds:
                lines = cv2.HoughLines(img_gray, 1, np.pi / 180, threshold)
                if lines is None:
                    pass
                else:
                    thetas = []
                    for line in lines:
                        rho, theta = line[0]
                        if 0 < theta < 3.14:
                            thetas.append(theta)
                    if len(thetas) > 0:
                        average_theta = sum(thetas) / len(thetas)
                        y_range = 450.0 if count_variable[self.label] == 1 else 150.0
                        reading = (average_theta - math.pi / 2) / (math.pi / 2) * y_range
                        self.result = abs(reading)
                        self.result = str(round(self.result, 2))
                    break
        print("图像中检测到 {} —— {}, 其读数为 {}".format(self.label, CN_names_param[self.label], self.result))
        self.write_json(count_variable[self.label], x, y, w, h, index)
        count_variable[self.label] += 1

    # 扳手阀门、开关和旋钮的角度识别，从而判断上述类别的状态
    def switch_angle_judge(self, name, count, index, count_variable):
        ellipse = cv2.fitEllipse(self.contour)
        angle = ellipse[2]
        x, y, w, h = self.bbox
        mmcv.imwrite(self.crop, '{}/{}_{}.png'.format(self.now_crop_save_dir, name, count))
        if self.label == 'f2':
            if 0 < angle <= 70 or 110 < angle < 180:
                self.result = "开"
            elif 70 < angle <= 110:
                self.result = "关"
        elif self.label == 'k':
            if 0 < angle <= 70:
                self.result = "开"
            elif 70 < angle < 90:
                self.result = "关"
        elif self.label == 'x':
            if 0 <= angle < 20 or 160 < angle <= 180:
                self.result = '中'
            elif 20 <= angle < 90:
                self.result = '右'
            elif 90 < angle <= 160:
                self.result = '左'
        print("图像中检测到 {} —— {}, 其状态为 {}".format(self.label, CN_names_param[self.label], self.result))
        self.write_json(count_variable[self.label], x, y, w, h, index)
        count_variable[self.label] += 1

    # 后处理过程
    def post_processing(self, name, i, count_variable):
        index = name  # 点位编码
        self.now_crop_save_dir = os.path.join(self.crop_save_dir, self.label)
        if not os.path.exists(self.now_crop_save_dir):
            os.makedirs(self.now_crop_save_dir)
        if '{}_{}'.format(EN_name_param[self.label], count_variable[self.label]) in self.json_info[index]["content"].keys() and \
                CN_names_param[self.label] in self.json_info[index]['content']['{}_{}'.format(EN_name_param[self.label], count_variable[self.label])].values():
            if self.label in ['xf1', 'xf2', 'xf3', 'xf4', 'xf5', 'm', 't', 'b']:  # 消防设施
                self.fire_equipment_detect(name, i, index, count_variable)
            elif self.label in ['aq', 'r']:  # 人和安全帽
                print()

            elif self.label == 'a':  # 按钮
                self.judge_button_color(name, i, index, count_variable)

            elif self.label in ['p', 'y1']:  # 标签、铭牌和数字仪表
                self.recognize_OCR(name, i, index, count_variable)

            elif self.label == 'f':  # 竖向阀门
                self.shu_famen_recognize(name, i, index, count_variable)

            elif self.label == 'f1':  # 横向阀门
                self.heng_famen_recognize(name, i, index, count_variable)

            elif self.label == 's':  # 水表
                self.recognize_round_meter(name, i, index, count_variable)

            elif self.label == 'y':  # 仪表
                self.recognize_square_meter(name, i, index, count_variable)

            elif self.label in ['f2', 'k', 'x']:  # 扳手阀门、开关和旋钮
                self.switch_angle_judge(name, i, index, count_variable)
            else:
                print("暂时无法对类别 {} 进行后处理！！".format(self.label))
        else:
            print("该图片中没有或检测到过多的 {} 类别！！".format(self.label))

    # 保存结果
    def save_result(self, name):
        if self.save_dir is None:
            return self.img_show
        else:
            mmcv.imwrite(self.img_show, os.path.join(self.whole_save_dir, name + '.jpg'))
        all_json_file = open(os.path.join(self.json_save_dir, 'defJson.json'), 'w')
        json.dump(self.json_info, all_json_file, ensure_ascii=False, indent=2)  # 保存所有json文件

        json_file = open(os.path.join(self.json_save_dir, name + '.json'), 'w')
        json.dump(self.json_info[name[-8:-4]], json_file, ensure_ascii=False, indent=2)  # 保存单个json文件

    # 机器人巡检主函数
    def forward(self, frame, name):
        index = name[-8:-4]  # 点位编码
        count_variable = {'a': 1, 'aq': 1, 'b': 1, 'f': 1, 'f1': 1, 'f2': 1, 'k': 1, 'm': 1,
                          'p': 1, 'r': 1, 's': 1, 't': 1, 'x': 1, 'xf1': 1, 'xf2': 1, 'xf3': 1,
                          'xf4': 1, 'xf5': 1, 'y': 1, 'y1': 1}
        self.SOLO_detect_result(self.SOLO_model, frame, index)  # SOLO检测结果
        if self.detect_results:
            tmp = self.object_sort(count_variable)  # 检测结果排序
            i = 0
            for detect_results in tmp.values():
                for detect_result in detect_results:  # 遍历每一个检测结果
                    self.label = detect_result[0]  # 类别
                    self.bbox = detect_result[1]  # 检测矩形框
                    self.mask = detect_result[2]  # mask
                    self.crop = detect_result[3]  # 裁剪小图
                    self.contour = detect_result[4]  # 部件轮廓
                    self.post_processing(index, i, count_variable)  # 后处理
                    self.save_result(name)  # 保存结果
                    i += 1
            self.detect_results = []  # 清空

            return self.json_info[index]
        else:
            mmcv.imwrite(frame, os.path.join(self.whole_save_dir, name + '.jpg'))
            return None
