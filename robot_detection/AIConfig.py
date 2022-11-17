
# 模型及配置文件相关路径
AI_weight = {
    'config_file': './solov2_DCN_robot_detection.py',  # 配置文件路径
    'checkpoint_file': './latest.pth'  # 模型路径
}

# 测试相关路径
test = {
    'json_path': '../defJson.json',  # 默认json路径
    'img_dir': '/home/wy/data/robot_detection/test/',  # 检测图片路径
    'save_dir': './test/',  # 结果保存路径
    'log_path': './log'  # 日志保存路径
}

# 类别对应的英文名
EN_name_param = {
    'a': 'Button', 'aq': 'Helmet', 'b': 'Label', 'f': 'Vvalue', 'f1': 'Hvalue', 'f2': 'Wvalue', 'k': 'Switch',
    'm': 'Extinguisher', 'p': 'Nameplate', 'r': 'Person', 's': 'Dials', 't': 'Shovel', 'x': 'Knob',
    'xf1': 'FBucket', 'xf2': 'FHook', 'xf3': 'Faxe', 'xf4': 'Fsandbox', 'xf5': 'Fsandbag', 'y': 'Meter',
    'y1': 'DigitalLED'
}

# 类别对应的中文名
CN_names_param = {
    'a': '按钮', 'aq': '安全帽', 'b': '标签卡', 'f': '竖向阀门', 'f1': '横向阀门', 'f2': '扳手阀门', 'k': '开关',
    'm': '灭火器', 'p': '铭牌', 'r': '人', 's': '水表', 't': '铁锹', 'x': '旋钮', 'xf1': '消防桶', 'xf2': '消防钩',
    'xf3': '消防斧头', 'xf4': '消防沙箱', 'xf5': '消防沙袋', 'y': '仪表', 'y1': '数字仪表'
}

# 类别对应type
type_param = {
    'a': 13, 'aq': 19, 'b': 12, 'f': 15, 'f1': 16, 'f2': 17, 'k': 18, 'm': 7, 'p': 11,
    'r': 20, 's': 8, 't': 1, 'x': 14, 'xf1': 2, 'xf2': 3, 'xf3': 4, 'xf4': 5, 'xf5': 6,
    'y': 9, 'y1': 10
}

# 水表对应起点
meter_start = {
    '11_3': 135, '12_3': {'1': 45, '2': 10}, '13_4': 45, '21_3': 45, '22_4': {'1': 100, '2': 45},
    '23_4': 45, '31_3': 45, '32_3': {'1': 45, '2': -80}, '32_4': 50, '41_3': -135, '41_4': 45, '51_4': 45,
    '52_3': 45, '52_4': 50, '54_3': 45
}  # 32_3_2, 41_3按照角度相加的原则进行计算

# 水表对应量程
meter_range = {
    '11_3': 1.6, '12_3': {'1': 1.6, '2': 100}, '13_4': 1.6, '21_3': 1.6, '22_4': {'1': 1.6, '2': 100},
    '23_4': 1.6, '31_3': 1.6, '32_3': {'1': 1.6, '2': 100}, '32_4': 1.5, '41_3': 1.5, '41_4': 1.6, '51_4': 2.5,
    '52_3': 2.5, '52_4': 1.6, '54_3': 2.5
}

# 每一类的状态信息获取算法
func_list = {'a': 'judge_button_color', 'aq': 'fire_detect', 'b': 'fire_detect', 'f': 'shu_famen_recognize',
             'f1': 'heng_famen_recognize', 'f2': 'angle_judge', 'k': 'angle_judge', 'm': 'fire_detect',
             'p': 'recognize_OCR', 'r': 'fire_detect', 's': 'recognize_round_meter', 't': 'fire_detect',
             'x': 'angle_judge', 'xf1': 'fire_detect', 'xf2': 'fire_detect', 'xf3': 'fire_detect',
             'xf4': 'fire_detect', 'xf5': 'fire_detect', 'y': 'recognize_spuare_meter', 'y1': 'recognize_OCR'}

# 阈值参数设置
Threshold_parameter = {
    'button': {
        'w': 30,
        'h': 30,
        'V': 128
    },
    'round_meter': {
        'line': [25, 20, 15, 10]
    },
    'square_meter': {
        'line': [23, 20, 15]
    }
}
