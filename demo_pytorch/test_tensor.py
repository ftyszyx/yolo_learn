import torch

# create a tensor --------------------------------
tensor = torch.tensor([1, 2, 3])
print(tensor)

# tensor properties --------------------------------
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)
print(tensor.dim())

# tensor operations --------------------------------
# 基本运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法
print(a + b)  # tensor([5, 7, 9])
print(torch.add(a, b))  # 同上

# 乘法
print(a * b)  # 元素级乘法
print(torch.mul(a, b))  # 同上

# 矩阵乘法
c = torch.tensor([[1, 2], [3, 4]])
d = torch.tensor([[5, 6], [7, 8]])
print(torch.matmul(c, d))  # 矩阵乘法

# get tensor data --------------------------------

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 获取单个元素
print(tensor[0, 0])  # 1

# 获取一行
print(tensor[1])  # tensor([4, 5, 6])

# 切片
print(tensor[0:2, 1:])  # 前两行，第二列往后

# 获取所有元素
print(tensor.flatten())  # tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

# change tensor shape --------------------------------
print("change tensor shape --------------------------------")
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 改变形状
reshaped = tensor.reshape(3, 2)
print(reshaped)

# 转置
transposed = tensor.t()
print(transposed)

# 添加维度
unsqueezed = tensor.unsqueeze(0)  # 在第0维添加维度
print(unsqueezed.shape)
print(unsqueezed)
