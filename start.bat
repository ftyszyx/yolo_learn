@echo off
echo 准备数据集...
python prepare_data.py

echo 开始训练模型...
python train.py

echo 训练完成！
pause

