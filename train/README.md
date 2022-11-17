# [实例分割SOLO系列](https://github.com/WXinlong/SOLO)

## 论文

    SOLO: Segmenting Objects by Locations,
    Xinlong Wang, Tao Kong, Chunhua Shen, Yuning Jiang, Lei Li
    In: Proc. European Conference on Computer Vision (ECCV), 2020 arXiv preprint (https://arxiv.org/abs/1912.04488)

    SOLOv2: Dynamic and Fast Instance Segmentation,
    Xinlong Wang, Rufeng Zhang, Tao Kong, Lei Li, Chunhua Shen
    In: Proc. Advances in Neural Information Processing Systems (NeurIPS), 2020
    arXiv preprint (https://arxiv.org/abs/2003.10152)

## 环境配置

### 1. 安装SOLO

    a.创建虚拟环境并激活
        conda create -n solo python=3.7 -y
        conda activate solo
    b.安装PyTorch、torchvision
        pip install torch==1.4.0
        pip install torchvision==0.5.0
        pip install cython
    c.下载SOLO和mmcv(若已下载，可忽略)
        git clone https://github.com/WXinlong/SOLO.git
        cd SOLO
    d.安装构建要求，然后安装SOLO
        pip install -r requirements/build.txt
        pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
        pip install -v -e .  # or "python setup.py develop"
        # mmcv手动安装可省略，上述pip install -v -e .命令中已经安装mmcv，在requirements/runtime.txt
        ################################################
        任选一种方式安装mmcv-0.2.16，推荐选择b)，速度较快
        a)  git clone https://github.com/open-mmlab/mmcv.git
            cd mmcv
            pip install -e .
        b)  pip install mmcv==0.2.16
        ################################################
    返回Successfully installed Pillow-6.2.2 mmdet-1.0.0+unknown scipy-1.7.0 terminaltables-3.1.0
    Removed build tracker: '/tmp/pip-req-tracker-ywswbqnb'即为安装成功。

### 2. 数据集路径

        SOLO
        ├── mmdet
        ├── tools
        ├── configs
        ├── data
        │   ├── coco
        │   │   ├── annotations
        │   │   ├── images
        │   │   │   ├── train2017
        │   │   │   ├── val2017
        │   │   │   ├── test2017
    
    总结--从零开始安装SOLO：
        conda create -n solo python=3.7 -y
        conda activate solo

        pip install torch==1.4.0
        pip install torchvision==0.5.0
        pip install cython
        git clone https://github.com/WXinlong/SOLO.git
        cd SOLO
        pip install -r requirements/build.txt
        pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
        pip install -v -e .

        # mmcv手动安装可省略，上述pip install -v -e .命令中已经安装mmcv，在requirements/runtime.txt
        ################################################
        任选一种方式安装mmcv-0.2.16，推荐选择b)，速度较快
        a)  git clone https://github.com/open-mmlab/mmcv.git
            cd mmcv
            pip install -e .
        b)  pip install mmcv==0.2.16
        ################################################

## 训练

测试环境是否安装成功：使用"demo/inference_demo.py"进行测试。

### 1. 数据集准备

    a.训练所需数据：图像以及对应的json文件，然后用labelme2coco.py将数据转换为coco数据集格式，数据集路径见上
    b.更改数据集：在mmdet/datasets/coco.py 中，注释掉原本的类别，更改为自己的类别：
        from .coco import CocoDataset
        from .registry import DATASETS
        
        @DATASETS.register_module
        class CocoDataset(CustomDataset):
            CLASSES = ['a', 'aq', 'b', 'f', 'f1', 'f2', 'k', 'm', 'p', 'r', 's', 't', 'x',
                    'xf1', 'xf2', 'xf3', 'xf4', 'xf5', 'y', 'y1']  # 改成自己的类别

### 2.配置文件修改：在configs/solo或solov2 下选择训练文件进行修改，例如solov2_light_512_dcn_r50_fpn_8gpu_3x.py

    25行：num_classes=21  # 类别数加1
    66行：dataset_type = 'CocoDataset'  # 数据集名称
    67行：data_root = 'data/coco/'  # 数据集根目录，根据情况进行更改
    74-75行、88行：可修改图片尺寸
    99行-116行：训练集和验证集路径，保证数据集读取路径正确
    118行：optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # 更改学习率lr=0.00125
    137行：total_epochs = 36  # 训练epoch
    138行：device_ids = range(1)  # 更改GPU数量
    141行：work_dir = './work_dirs/solov2_light_512_dcn_release_r50_fpn_8gpu_1x'  # 模型保存路径

### 3. 训练

```shell
# 单GPU训练
python tools/train.py ${CONFIG_FILE}

# 举例:
python tools/train.py configs/solo/solo_r50_fpn_8gpu_1x.py
```

## 测试

```shell
# 单GPU测试
python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm
# 举例: 
python tools/test_ins.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --out  results_solo.pkl --eval segm
```

    批量图片测试：
        更改demo/inference_demo.py 进行测试
    计算精度：
        python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm
        # 举例: 
        python tools/test_ins.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --out  results_solo.pkl --eval segm
        
    Tips：pycharm运行test_ins.py时，eval_type应默认写为['segm'] 
    上述程序只能计算模型的精度，不能计算每个类别的精度，如需要计算每个类别的精度，将"mmdet/core/evaluation/coco_utils.py"中coco_eval函数中的classwise改为True

## 代码分析

### loss计算过程

        网络正向传播得到cate_preds和ins_preds，以及真实数据的gt_bbox_list, gt_label_list和gt_mask_list。
        需要明确：cate_preds和ins_preds是五个feature map经过anchor head得到的结果，相当于五个尺度concat起来得到的一个list；而gt中的真实数据是基于整个img的各种标注。
        步骤：
        （1）solo_target_single函数，其目的是根据scale_range，将所有gt instance分配到五个不同的不同level的特征图上（通过计算instance bbox的面积和scale_range做比较，以确定该instance落在那个level上）；
        （2）solo_target_single函数逻辑：计算出每个instance的bbox大小，将其分配到对应scale_range的level上，将gt放缩到该level特征图的大小，
            从而得到instance mask质心所对应的grid的索引，进而可以确定哪些grid负责预测该instance（正例），最后gt的形式就可以拆分成五个level的组合；
        （3）loss函数，根据正例索引，筛选出所有正例grid所对应的mask（gt和pred都做筛选），将pred mask做sigmoid归一化以后，两者计算Dice Loss，作为分割损失；
        （4）将gt cate_label转化为[3872]维向量，将pred cate_label转化为[3872, 80]维矩阵，两者做Focal Loss；
        （5）返回Dice Loss和Focal Loss的值，传到上述的"mmdet/api/train.py"中parse_losses函数中计算最终的loss值。
    训练大致过程：
        "tools/train.py"中model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)创建模型；
        "mmdet/apis/train.py"根据是否需要分布式训练，将数据放在GPU上，对数据、模型、训练策略、评估、推理进行融合；
        "mmcv/runner/runner.py"根据配置文件确定train和val的情况，通过run(),train(),val(),rescue(),save_checkpoint()等函数进行训练；
        每一个epoch调用"tools/train.py"中batch_processor()函数，输出output即为loss；
        batch_processor()中model(∗∗data)是最重要的一个环节，将当前batch作为输入传递到"mmdet/models/detectors/single_stage_ins.py"中，
            调用forward_train()函数得到预测结果outs和mask_head_pred，加上batch中gt部分作为输入；
            (其中用到"mmdet/models/backbone/resnet.py"、"mmdet/models/neck/fpn.py"、"mmdet/models/anchor_heads/solo_head.py"、"mmdet/models/mask_heads/mask_feat_head.py"来获得预测结果)
        传递到"mmdet/models/anchor_heads/solo_head.py"中loss()函数计算损失，计算过程如上所述，根据hook的一系列操作进行梯度下降方向传播，来做网络参数的更新。
    推理大致过程（以demo为例）：
        "demo/inference_demo.py"中调用init_detector()创建model，inference_detector()做正向inference，最后show结果；
        "mmdet/apis/inferece.py"中init_detector函数，其核心函数是build_detector，根据config文件信息创建模型，加载checkpoint；inference_detector()函数首先进行augmentation，然后调用model做inference；
        "mmdet/model/builder.py"中build_detector()函数调用"mmdet/utils/registry.py"中build_from_cfg()函数，通过config中的字典来对模型进行搭建，obj_cls就是要创建的module，如SOLO，ResNet，FPN等等，
            只有某个注册器中有配置文件中存在的type时，才会对该注册器进行register，通过args中的dict得到相应的module；
        根据type找到对应文件"mmdet/models/detectors/solo.py"对应的"mmdet/models/detectors/single_stage_ins.py",调用simple_test()进行inference，通过特征提取和处理得到ins_pred、cate_pred，
            经过get_seg()函数生成masks，然后经过Matrix NMS方法进行非极大抑制，输出最后的结果。
        