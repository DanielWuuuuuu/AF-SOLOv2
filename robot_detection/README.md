# 智能巡检识别演示

## 1. AF-SOLOv2模型[下载](https://pan.baidu.com/s/1FBb2hpq3mNROEVjSVPy6ZA?pwd=7668)、测试数据[下载](https://pan.baidu.com/s/18m4opUIBw0mS7r4fzxIkhg?pwd=1ed1)、PaddlePaddleOCR模型[下载](https://pan.baidu.com/s/1aD9vXvwKyHKpySgZcHJv-A?pwd=001v)

## 2. 程序说明

>robot_detection_api.py：项目接口程序  
>demo_robot_detection.py：调用项目接口demo程序  
>AIConfig.py：各种配置文件路径以及参数程序  
>Tools.py：工具类相关函数程序

    如若对小目标精度要求较高，配置文件可使用configs/solov2/solov2_DCN_robot_detection.py

## 3. 运行

配置设置完成后，运行即可

```shell
python demo_robot_detection.py
```
