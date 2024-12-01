@echo off

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

@REM echo Installing CUDA PyTorch...
@REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://mirrors.aliyun.com/pypi/simple/


@REM echo Installing other requirements...
@REM pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

echo Verifying GPU installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

pause