import torch
import torch.onnx

def convert_to_onnx(model, input_size, save_path):
    """
    将PyTorch模型转换为ONNX格式
    
    Args:
        model: PyTorch模型
        input_size: 输入张量的大小，例如 (1, 3, 224, 224)
        save_path: ONNX模型保存路径
    """
    # 设置模型为评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_size)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,                     # 要转换的模型
        dummy_input,              # 模型输入
        save_path,                # 保存路径
        export_params=True,       # 存储训练好的参数权重
        opset_version=11,         # ONNX算子集版本
        do_constant_folding=True, # 是否执行常量折叠优化
        input_names=['input'],    # 输入节点的名称
        output_names=['output'],  # 输出节点的名称
        dynamic_axes={            # 动态尺寸设置
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"模型已成功导出到: {save_path}")

# 使用示例
if __name__ == "__main__":
    # 假设您有一个预训练的PyTorch模型
    # model = YourModel()
    # model.load_state_dict(torch.load('your_model.pth'))
    
    # 设置输入大小和保存路径
    # input_size = (1, 3, 224, 224)  # (batch_size, channels, height, width)
    # save_path = "model.onnx"
    
    # 转换模型
    # convert_to_onnx(model, input_size, save_path)
