@echo off
echo 检查GPU状态...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo 检查数据集...
python check_dataset.py

echo 准备数据集...
python prepare_data.py

echo 开始训练模型...
python train.py

echo 训练完成！
pause

