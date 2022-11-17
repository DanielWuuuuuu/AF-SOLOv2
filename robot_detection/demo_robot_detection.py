from robot_detection_api import Load_Model, Robot_Detection
from AIConfig import *
import time
import os
import cv2
import json
import sys
sys.path.append('../mmdet')
sys.path.append('../')


t1 = time.time()
json_file = open(test['json_path'], encoding='utf-8')
json_info = json.load(json_file)
robot_detection = Robot_Detection(json_info)
t2 = time.time()
print('预处理所花时间为：{} 秒'.format(round(t2 - t1, 4)))
# 检测
all_length = 0  # 检测图片数量
for img_name in os.listdir(test['img_dir']):
    print("***********************检测的图片名为：{} ***********************".format(img_name))
    img = cv2.imread(os.path.join(test['img_dir'], img_name))
    # 显示检测结果
    t3 = time.time()
    jsonData = robot_detection.forward(img, img_name)
    print("检测结果转为json格式为：\n", jsonData)
    t4 = time.time()
    print('--------------------检测 {} 所花时间为：{} 秒--------------------\n'.format(img_name, round(t4 - t3, 4)))
    all_length += 1
t5 = time.time()
FPS = all_length / round(t5 - t2, 4)
print("总共检测 {} 张图片\t\t所花时间：{} 秒\t\t当前帧率FPS：{}".format(all_length, round(t5 - t2, 4), round(FPS, 2)))

