# about this project

this project is fork from ultraylitics: https://github.com/ultralytics/ultralytics
the purpose is to learn the ultralytics code

# begin

environment:

- python 3.10

## install environment

```
conda create -n yolo python=3.10
```

```
conda activate yolo
```

<!-- install yolo as edit mode(源码安装)

```
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
``` -->

install cuda support 

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

install yolo

```
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## run test

```
python tools/test_cuda.py
```


## export rknn
需要先将pt转成onnx（用https://github.com/airockchip/ultralytics_yolo11.git）去转
再将onnx转成krnn.
https://blog.csdn.net/zhangqian_1/article/details/142722526.


## rknn tookit安装
需要linux环境
https://docs.radxa.com/rock5/rock5c/app-development/rknn_install
参考这个导出rknn
https://docs.radxa.com/zero/zero3/app-development/rknn_ultralytics

