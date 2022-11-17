English | [简体中文](README_CN.md)

# [Inspection and identification algorithm based on improved SOLOv2 of intelligent robot in complex environment]()

On the basis of [SOLOv2](https://arxiv.org/abs/2003.10152), it is improved and optimized according to the needs of intelligent inspection robots. It can identify 15 types of equipment components and perform corresponding post-processing, with an accuracy of more than 90%.

This code of this project is mainly modified and summarized based on SOLO, if there is any error, please correct me! ! !

## Highlights

- **Multi-category part recognition in complex scenes:** For complex distribution station scenarios, up to 15 types of equipment components can be identified, and status information can be obtained for each type of equipment components;
- **Small object part recognition:** By optimizing the feature output of the feature pyramid, the recognition accuracy of small targets is improved, and the overall accuracy is slightly improved.

## Installation

This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v1.0.0).The installation method is as follows, also can refer to the original [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher (>=1.5 is not tested)
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv 0.2.16](https://github.com/open-mmlab/mmcv/tree/v0.2.16)

The software and hardware versions implemented in this project are as follows:

- OS: Ubuntu 18.04
- CUDA: 10.2.89
- CUDNN: 7.6.5
- NCCL: 2.8.3
- GCC(G++): 7.5.0

### Install SOLO

a. Create a conda virtual environment and activate it.

```shell
conda create -n solo python=3.7 -y
conda activate solo
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
pip install torch==1.4.0 torchvision==0.5.0
pip install cython
```

c. Clone the SOLO repository.

```shell
git clone https://github.com/WXinlong/SOLO.git
cd SOLO
```

d. Install build requirements and then install SOLO.

```shell
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
```

e. The SOLO environment is compiled and the GPU version of paddlepaddle is installed.

```shell
pip install paddlepaddle-gpu==2.0.1
```

## Usage

### A quick demo

Once the installation is done, you can download the provided [models](https://cloudstor.aarnet.edu.au/plus/s/dXz11J672ax0Z1Q/download) and use [inference_demo.py](demo/inference_demo.py) to run a quick demo.

### Train SOLO

```shell
# Train with single GPU
python tools/train.py ${CONFIG_FILE}

# Example
python tools/train.py configs/solo/solo_r50_fpn_8gpu_1x.py
```

### Testing SOLO

```shell
# single-gpu testing
python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm

# Example
python tools/test_ins.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --out  results_solo.pkl --eval segm
```

## Demonstration of Intelligent Patrol Inspection and Recognition

Please refer to [README.md](robot_detection/README.md)
