# -*- coding: utf-8 -*-
# @Time    : 2022-10-06 9:12
# @Author  : Wu You
# @File    : tools.py.py
# @Software: PyCharm


from PIL import Image, ImageDraw, ImageFont
from AIConfig import *
from functools import wraps
import numpy as np
import time
import cv2
import logging
from logging.handlers import RotatingFileHandler


def func_timer(function):
    """
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    """

    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name=function.__name__, time=t1 - t0))
        return result

    return function_timer


def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "../PaddleOCR/fonts/simfang.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def set_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = RotatingFileHandler("{}/{}.log".format(test['log_path'], filename), maxBytes=10 * 1024 * 1024, backupCount=10)

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger