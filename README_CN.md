[English](README.md) | 简体中文

# [基于改进SOLOv2的复杂场景下智能机器人巡检识别算法]()

在[SOLOv2](https://arxiv.org/abs/2003.10152)的基础上，根据智能巡检机器人需求进行改进优化，可识别15种类的设备部件并进行相应的后处理，精度在90%以上。

该项目代码主要是基于[SOLO](https://github.com/WXinlong/SOLO)进行更改与总结，如有错误，还请指正！！！

## 亮点

- **复杂场景下多类别部件识别:** 针对复杂的配电站场景，可识别高达15类的设备部件，并对每一类设备部件进行状态信息获取；
- **小目标部件识别:** 通过优化特征金字塔的特征输出，提高小目标的识别精度，整体精度略微提升。

## 安装

该实现基于[mmdetection](https://github.com/open-mmlab/mmdetection)(v1.0.0)。安装方式如下，也参阅原始[INSTALL.md](docs/INSTALL.md)进行安装和数据集准备。

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher (>=1.5 is not tested)
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv 0.2.16](https://github.com/open-mmlab/mmcv/tree/v0.2.16)

本项目实现的软硬件版本如下:

- OS: Ubuntu 18.04
- CUDA: 10.2.89
- CUDNN: 7.6.5
- NCCL: 2.8.3
- GCC(G++): 7.5.0

### 安装 SOLO

a. Conda创建虚拟环境并激活

```shell
conda create -n solo python=3.7 -y
conda activate solo
```

b. 按照[官方](https://pytorch.org/)安装PyTorch和torchvision

```shell
pip install torch==1.4.0 torchvision==0.5.0
pip install cython
```

c. 将SOLO项目文件下载到本地

```shell
git clone https://github.com/WXinlong/SOLO.git
cd SOLO
```

d. 安装编译环境并安装SOLO

```shell
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
```

e. SOLO环境编译完成，安装GPU版paddlepaddle

```shell
pip install paddlepaddle-gpu==2.0.1
```

## 使用

### 快速演示

安装完成后，可以下载提供的[模型](https://cloudstor.aarnet.edu.au/plus/s/dXz11J672ax0Z1Q/download)并运行[inference_demo.py](demo/inference_demo.py)进行快速演示。

### SOLO训练

```shell
# 单GPU训练
python tools/train.py ${CONFIG_FILE}

# 举例:
python tools/train.py configs/solo/solo_r50_fpn_8gpu_1x.py
```

### SOLO测试

```shell
# 单GPU测试
python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm

# 举例: 
python tools/test_ins.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --out  results_solo.pkl --eval segm
```

## 智能巡检识别演示

请参考[README.md](robot_detection/README.md)
