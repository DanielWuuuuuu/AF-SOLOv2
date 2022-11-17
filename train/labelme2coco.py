import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split

np.random.seed(41)

# 0为背景
classname_to_id = {"a": 1, "b": 2, "f": 3, "f1": 4, "f2": 5,
                   "m": 6, "p": 7, "s": 8, "t": 9, "x": 10,
                   "xf1": 11, "xf2": 12, "xf3": 13, "xf4": 14, "xf5": 15,
                   "y": 16, "y1": 17}


class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 1

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {'info': 'spytensor created', 'license': ['license'], 'images': self.images,
                    'annotations': self.annotations, 'categories': self.categories}
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {'id': v, 'name': k}
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image = {'height': h, 'width': w, 'id': self.img_id,
                 'file_name': os.path.basename(path).replace(".json", ".jpg")}
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {'id': self.ann_id, 'image_id': self.img_id, 'category_id': int(classname_to_id[label]),
                      'segmentation': [np.asarray(points).flatten().tolist()], 'bbox': self._get_box(points),
                      'iscrowd': 0, 'area': 1.0}
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    # 转换数据集的类型，train or test
    data_type = 'test'
    # 数据集路径
    train_data_path = "./data/robot_detection/new/train/"
    test_data_path = "./data/robot_detection/1"
    # 数据集保存路径
    saved_coco_path = "../data/"
    # 创建文件
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    if not os.path.exists("%scoco/train2017/" % saved_coco_path):
        os.makedirs("%scoco/train2017" % saved_coco_path)
    if not os.path.exists("%scoco/val2017/" % saved_coco_path):
        os.makedirs("%scoco/val2017" % saved_coco_path)
    if not os.path.exists("%scoco/test2017/" % saved_coco_path):
        os.makedirs("%scoco/test2017" % saved_coco_path)

    if data_type == 'train':
        # 获取images目录下所有的json文件列表
        json_list_path = glob.glob(train_data_path + "/*.json")
        # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
        train_path, val_path = train_test_split(json_list_path, test_size=0.1)
        print("train_n:", len(train_path), 'val_n:', len(val_path))

        # 把训练集转化为COCO的json格式
        l2c_train = Lableme2CoCo()
        train_instance = l2c_train.to_coco(train_path)
        l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
        for file in train_path:
            shutil.copy(file.replace("json", "jpg"), "%scoco/train2017/" % saved_coco_path)

        # 把验证集转化为COCO的json格式
        l2c_val = Lableme2CoCo()
        val_instance = l2c_val.to_coco(val_path)
        l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)
        for file in val_path:
            shutil.copy(file.replace("json", "jpg"), "%scoco/val2017/" % saved_coco_path)

    # 测试集
    if data_type == 'test':
        json_list_path = glob.glob(test_data_path + "/*.json")
        test_path = json_list_path
        print("test_n:", len(test_path))

        # 把测试集转化为COCO的json格式
        l2c_test = Lableme2CoCo()
        test_instance = l2c_test.to_coco(test_path)
        l2c_test.save_coco_json(test_instance, '%scoco/annotations/instances_test2017.json' % saved_coco_path)
        for file in test_path:
            shutil.copy(file.replace("json", "jpg"), "%scoco/test2017/" % saved_coco_path)
