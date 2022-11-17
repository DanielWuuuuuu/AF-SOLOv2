from mmdet.apis import init_detector, inference_detector, show_result_ins
import mmcv
import os
import time

base_path = "/home/wy/SOLO-master"  # 根路径

config_file = './solov2_DCN_robot_detection.py'  # 配置文件路径  './mask_rcnn_robot_detection.py'
checkpoint_file = './model_DCN_960*540_comparison/epoch_36.pth'  # 模型路径  './mask_rcnn_model/epoch_36.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print(model)

input_path = "{}/data/test".format(base_path)  # 输入图片路径
save_path = "./results/{}_results".format(os.path.basename(input_path))  # 保存路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# test images
all_start = time.time()
for img_name in os.listdir(input_path):
    img = mmcv.imread(os.path.join(input_path, img_name))
    save_file = os.path.join(save_path, img_name)

    start = time.time()
    result = inference_detector(model, img)  # 检测图片
    show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file=save_file)
    end = time.time()
    print("Detect image: {}\t\tTime is {}s!".format(img_name, end - start))
all_end = time.time()
all_length = len(os.listdir(input_path))
FPS = len(os.listdir(input_path)) / (all_end - all_start)
FPS = round(FPS, 2)
print("总共检测{}张图片\t\t所花时间：{}s\t\t当前帧率FPS：{}".format(all_length, all_end - all_start, FPS))
