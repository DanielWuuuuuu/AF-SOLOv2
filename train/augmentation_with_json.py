# -*- coding: utf-8 -*-
# @Time    : 2021-05-29 9:14
# @Author  : Wu You
# @File    : augmentation_with_json.py
# @Software: PyCharm
##############################################################

# 包括:
#     1. 改变亮度
#     2. 加噪声
#     3. 加随机点
#     4. 镜像(需要改变points)

import time
import random
import cv2
import os
import numpy as np
from skimage.util import random_noise
import base64
import json
import re
from copy import deepcopy
import argparse
import math


# 图像均为cv2读取
class DataAugmentForObjectDetection:
    def __init__(self, rotation_rate=0.5, max_rotation_angle=180, change_light_rate=0.5, change_color_rate=0.5,
                 add_noise_rate=0.5, random_point=0.5, flip_rate=0.5, shift_rate=0.5, scale_rate=0.5,
                 rand_point_percent=0.03, is_addNoise=True, is_changeLight=True, is_changeColor=True,
                 is_random_point=True, is_shift_pic_bboxes=True,
                 is_filp_pic_bboxes=True, is_rotate_img_bbox=True, is_scale_change=True):
        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.change_light_rate = change_light_rate
        self.change_color_rate = change_color_rate
        self.add_noise_rate = add_noise_rate
        self.random_point = random_point
        self.flip_rate = flip_rate
        self.shift_rate = shift_rate
        self.scale_rate = scale_rate

        self.rand_point_percent = rand_point_percent

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_changeColor = is_changeColor
        self.is_random_point = is_random_point
        self.is_filp_pic_bboxes = is_filp_pic_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_scale_change = is_scale_change

    # 旋转
    def rotate_img_bbox(self, img, json_info, angle=5, scale=1.):
        """
        输入:
            img:图像array,(h,w,c)
            json_info:json文件的数据信息
            angle:旋转角度（角度制）
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        """
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # 计算新图片的宽和高
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # 利用opencv得到旋转矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # 结合旋转计算从旧中心到新中心的移动
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                                 flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正points坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取points里面的每一个点的坐标，然后求旋转之后的对应点的坐标
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                point = np.dot(rot_mat, np.array([p[0], p[1], 1]))
                p[0] = point[0]
                p[1] = point[1]
        return rot_img, json_info

    # 加噪声
    def addNoise(self, img):
        return random_noise(img, seed=int(time.time())) * 255

    # 调整亮度
    def changeLight(self, img):
        alpha = random.uniform(0.9, 0.99)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # 调整颜色
    def changeColor(self, image):
        I_param = random.uniform(0.9, 1.1)  # ##(0.7, 5)这个范围如果修改需要做实验验证
        S_param = random.uniform(0.9, 1.1)  # ##这个范围如果修改需要做实验验证
        HSI = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2HLS)
        (H, S, I) = cv2.split(HSI.astype("float"))

        I2 = I * I_param
        # I2 = I
        S2 = S * S_param

        merged = cv2.merge([H, S2, I2])
        merged2 = merged.astype("uint8")
        RGB = cv2.cvtColor(merged2, cv2.COLOR_HLS2BGR)

        return RGB

    # 随机的改变点的值
    def addRandPoint(self, img):
        percent = self.rand_point_percent
        num = int(percent * img.shape[0] * img.shape[1])
        for i in range(num):
            rand_x = random.randint(0, img.shape[0] - 1)
            rand_y = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[rand_x, rand_y] = 0
            else:
                img[rand_x, rand_y] = 255
        return img

    # 平移
    def shift_pic_bboxes(self, img, json_info):

        # ---------------------- 平移图像 ----------------------
        h, w, _ = img.shape
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        shapes = json_info['shapes']
        for shape in shapes:
            points = np.array(shape['points'])
            x_min = min(x_min, points[:, 0].min())
            y_min = min(y_min, points[:, 1].min())
            x_max = max(x_max, points[:, 0].max())
            y_max = max(y_max, points[:, 1].max())

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移points ----------------------
        for shape in shapes:
            for p in shape['points']:
                p[0] += x
                p[1] += y
        return shift_img, json_info

    # 镜像
    def filp_pic_bboxes(self, img, json_info):

        # ---------------------- 翻转图像 ----------------------
        h, w, _ = img.shape

        sed = random.random()

        if 0 < sed < 0.33:  # 0.33的概率水平翻转，0.33的概率垂直翻转,0.33是对角反转
            flip_img = cv2.flip(img, 0)  # _flip_x
            inver = 0
        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(img, 1)  # _flip_y
            inver = 1
        else:
            flip_img = cv2.flip(img, -1)  # flip_x_y
            inver = -1

        # ---------------------- 调整points ----------------------
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                if inver == 0:
                    p[1] = h - p[1]
                elif inver == 1:
                    p[0] = w - p[0]
                elif inver == -1:
                    p[0] = w - p[0]
                    p[1] = h - p[1]

        return flip_img, json_info

    # 缩放
    def scale_change(self, img, json_info):
        scale_change_rate = random.uniform(0.5, 2)
        change_img = cv2.resize(img, (int(img.shape[1] // 2 * scale_change_rate), int(img.shape[0] // 2 * scale_change_rate)))

        # ---------------------- 缩放points ----------------------
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                p[0] = p[0] // 2 * scale_change_rate
                p[1] = p[1] // 2 * scale_change_rate
        json_info["imageHeight"] = int(img.shape[0] // 2 * scale_change_rate)
        json_info["imageWidth"] = int(img.shape[1] // 2 * scale_change_rate)
        return change_img, json_info

    # 图像增强方法
    def dataAugment(self, img, dic_info):

        change_num = 0  # 改变的次数
        while change_num < 2:  # 默认至少有一种数据增强生效

            # if self.is_rotate_img_bbox:
            #     if random.random() > self.rotation_rate:  # 旋转
            #         change_num += 1
            #         angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            #         scale = random.uniform(0.7, 0.8)
            #         img, dic_info = self.rotate_img_bbox(img, dic_info, angle, scale)

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self.changeLight(img)

            # if self.is_changeColor:
            #     if random.random() > self.change_color_rate:  # 改变颜色
            #         change_num += 1
            #         img = self.changeColor(img)

            # if self.is_addNoise:
            #     if random.random() < self.add_noise_rate:  # 加噪声
            #         change_num += 1
            #         img = self.addNoise(img)

            # if self.is_random_point:
            #     if random.random() < self.random_point:  # 加随机点
            #         change_num += 1
            #         img = self.addRandPoint(img)

            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img, dic_info = self.shift_pic_bboxes(img, dic_info)

            if self.is_filp_pic_bboxes or 1:
                if random.random() < self.flip_rate:  # 翻转
                    change_num += 1
                    img, bboxes = self.filp_pic_bboxes(img, dic_info)

            # if self.is_scale_change:
            #     if random.random() < self.scale_rate:  # 缩放
            #         change_num += 1
            #         img, dic_info = self.scale_change(img, dic_info)

        return img, dic_info


# json解析工具
class ToolHelper:
    # 从json文件中提取原始标定的信息
    def parse_json(self, path):
        with open(path) as f:
            json_data = json.load(f)
        return json_data

    # 对图片进行字符编码
    def img2str(self, img_name):
        with open(img_name, "rb")as f:
            base64_data = str(base64.b64encode(f.read()))
        match_pattern = re.compile(r'b\'(.*)\'')
        base64_data = match_pattern.match(base64_data).group(1)
        return base64_data

    # 保存图片结果
    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)

    # 保持json结果
    def save_json(self, file_name, save_folder, dic_info):
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(dic_info, f, indent=2)


if __name__ == '__main__':

    need_aug_num = 5  # 每张图片需要增强的次数

    toolhelper = ToolHelper()  # 工具

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str, default="../")
    parser.add_argument('--save_img_path', type=str, default="../")
    parser.add_argument('--source_json_path', type=str, default="../")
    parser.add_argument('--save_json_path', type=str, default="../")
    args = parser.parse_args()
    source_img_path = args.source_img_path  # 图片文件原始位置
    save_img_path = args.save_img_path  # 图片增强结果保存文件
    source_json_path = args.source_json_path  # json文件原始位置
    save_json_path = args.save_json_path  # json增强结果保存文件

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    if not os.path.exists(save_json_path):
        os.mkdir(save_json_path)

    for parent, _, files in os.walk(source_img_path):
        files.sort()  # 排序一下
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                cnt = 0
                pic_path = os.path.join(parent, file)
                json_path = os.path.join(source_json_path, file[:-4] + '.json')
                json_dic = toolhelper.parse_json(json_path)
                # 找到文件的最后名字
                dot_index = file.rfind('.')
                _file_prefix = file[:dot_index]  # 文件名的前缀
                _file_suffix = file[dot_index:]  # 文件名的后缀
                img = cv2.imread(pic_path)

                while cnt < need_aug_num:  # 继续增强
                    auged_img, json_info = dataAug.dataAugment(deepcopy(img), deepcopy(json_dic))
                    img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                    img_save_path = os.path.join(save_img_path, img_name)
                    toolhelper.save_img(img_save_path, auged_img)  # 保存增强图片

                    json_info['imagePath'] = img_name
                    base64_data = toolhelper.img2str(img_save_path)
                    json_info['imageData'] = base64_data
                    toolhelper.save_json('{}_{}.json'.format(_file_prefix, cnt + 1),
                                         save_json_path, json_info)  # 保存json文件
                    print(img_name)
                    cnt += 1  # 继续增强下一张
