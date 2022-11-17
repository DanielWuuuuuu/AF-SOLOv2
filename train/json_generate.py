import numpy as np
import cv2
import base64
import re
import os
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import time
import json
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import math


# 生成imageData
def img2str(image_name):
    with open(image_name, "rb")as file:
        base64_data = str(base64.b64encode(file.read()))
    match_pattern = re.compile(r"b'(.*)'")
    base64_data = match_pattern.match(base64_data).group(1)
    return base64_data


def generate_points(image, label, new_shapes):
    # change_mask = np.where(image == 1, 255, 0)
    # skimage格式转为cv2格式
    # result = img_as_ubyte(change_mask).astype('uint8')

    # result = np.where(image == 1, 255, 0).astype('uint8')
    result = image
    # # #####分割图像的二值化处理###
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    [ret, th2] = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)
    th2 = cv2.dilate(th2, kernel)

    # #########画出边缘图像#######
    contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # ######针对每一个闭合的边缘图像####
    del list(contours)[0]
    for contour in contours:
        points = contour[::10]  # 每隔20个像素取一个点

        new_points = []
        for point in points:
            new_points.append(point[0].tolist())
        shape_data = {
            "label": label,
            "points": new_points,
            "group_id": None,
            "shape_type": 'polygon',
            "flags": {}
        }
        new_shapes.append(shape_data)
    return new_shapes


def write_json(shapes, masked_image, img_Data):
    # 写入json
    json_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.join(test_img_path, img_name),
        "imageData": img_Data,
        "imageHeight": masked_image.shape[0],
        "imageWidth": masked_image.shape[1]
    }
    # 保存json文件
    print(os.path.join(json_save_path, img_name[:-4] + '.json'))
    f = open(os.path.join(json_save_path, img_name[:-4] + '.json'), 'w')
    json.dump(json_data, f, indent=2)


# 检测并保存结果
def show_result_ins(img, result, class_names, score_thr=0.3, sort_by_density=False, out_file=None,
                    name=None):
    """Visualize the instance segmentation results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The instance segmentation result.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the masks.
        sort_by_density (bool): sort the masks by their density.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """

    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img_show = img.copy()  # 原图
    h, w, _ = img.shape

    crop_save_dir = save_dir + 'crop/'
    whole_save_dir = save_dir + 'whole/'
    json_save_dir = save_dir + 'json/'
    if not os.path.exists(json_save_dir):
        os.makedirs(json_save_dir)

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
    new_shapes = []
    for idx in range(num_mask):
        idx = -(idx+1)
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
        mask = mask * 255

        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]
        label_text = class_names[cur_cate]
        new_shapes = generate_points(mask, label_text, new_shapes)

        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(img_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
        i += 1
    if out_file is None:
        return img_show
    else:
        mmcv.imwrite(img_show, os.path.join(whole_save_dir, name))
        return new_shapes


if __name__ == "__main__":
    json_save_path = './json'
    # 配置文件路径
    config_file = './solov2_DCN_robot_detection.py'
    # 模型路径
    checkpoint_file = './model_DCN_960*540_add4/epoch_48.pth'
    # 检测图片路径
    test_img_path = './data/robot_detection/1000/'
    # 结果保存路径
    save_dir = './data/robot_detection/new_img_1000_result/'

    # 通过配置文件加载模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    for img_name in os.listdir(test_img_path):
        img = cv2.imread(os.path.join(test_img_path, img_name))
        # 检测结果
        result = inference_detector(model, img)
        # 显示检测结果
        shapes = show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file=save_dir, name=img_name)

        img_data = img2str(os.path.join(test_img_path, img_name))
        write_json(shapes, img, img_data)
