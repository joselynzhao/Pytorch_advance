#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/8/22 下午5:02
# @Author  : Joselynzhao
# @Email   : zhaojing17@forxmail.com
# @File    : 07.py
# @Software: PyCharm
# @Desc    : PyTorch之小试牛刀


# import numpy as np
# N, D_in, H,D_out = 64,1000,100,10
# # 创建随机输入和输出数据
# x = np.random.randn(N,D_in)
# y = np.random.randn(N,D_out)
# # 随机初始化权重
# w1 = np.random.randn(D_in,H)
# w2 = np.random.randn(H,D_out)
#
# learning_rate  = 1e-6
# for t in range(500):
#     # 前向传递：计算预测值y
#     h = x.dot(w1)
#     h_relu = np.maximum(h,0)
#     y_pred = h_relu.dot(w2)
#     # 计算和打印损失loss
#     loss = np.square(y_pred-y).sum() # 均方损失
#     print(t,loss)
#
#     # 反向传播，计算w1和w2对loss的梯度
#     grad_y_pred = 2.0 *(y_pred-y) #两倍loss
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()
#     grad_h[h<0]=0
#     grad_w1 = x.T.dot(grad_h)
#
#     # 更新权重
#     w1 -=learning_rate * grad_w1
#     w2 -=learning_rate * grad_w2

    #一会儿回来画图分析.


# import torch
# dtype = torch.float
# # device = torch.device('cpu')
# device = torch.device('cuda') #取消注释以在GPU上运行
# # N是批量大小; D_in是输入维度;
# # H是隐藏的维度; D_out是输出维度。
# N, D_in, H, D_out = 64, 1000, 100, 10
# #创建随机输入和输出数据
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
#
# # 随机初始化权重
# w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype)
#
# learning_rate = 1e-6
# for t in range(500):
#     # 前向传递：计算预测y
#     h = x.mm(w1)  # mm表示点成
#     h_relu = h.clamp(min=0) # relu函数
#     y_pred = h_relu.mm(w2)
#
#     # 计算和打印损失
#     loss = (y_pred - y).pow(2).sum().item() #．item是什么意思呢? 获取数值
#     print(t, loss)
#
#     # Backprop计算w1和w2相对于损耗的梯度
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.t().mm(grad_y_pred)
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()  #对应copy
#     grad_h[h < 0] = 0
#     grad_w1 = x.t().mm(grad_h)
#
#     # 使用梯度下降更新权重
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2
#
#     # 测试了速度,确实快很多.


'''下面我们使用PyTorch的Tensors和autograd来实现我们的两层的神经网络；我们不再需要手动执行网络的反向传播：'''
import  torch
# dtype  = torch.float
# device  = torch.device('cpu')
# # device = torch.device("cuda")
# # N是批量大小; D_in是输入维度;
# # H是隐藏的维度; D_out是输出维度。
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # 创建随机Tensors以保持输入和输出。
# # 设置requires_grad = False表示我们不需要计算渐变
# # 在向后传球期间对于这些Tensors。
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
#
# # 为权重创建随机Tensors。
# # 设置requires_grad = True表示我们想要计算渐变
# # 在向后传球期间尊重这些张贴。
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
#
# learning_rate = 1e-6
# for t in range(500):
#     # 前向传播：使用tensors上的操作计算预测值y;
#     # 由于w1和w2有requires_grad=True，涉及这些张量的操作将让PyTorch构建计算图，
#     # 从而允许自动计算梯度。由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
#     y_pred = x.mm(w1).clamp(min=0).mm(w2)
#
#     # 使用Tensors上的操作计算和打印丢失。
#     # loss是一个形状为()的张量
#     # loss.item() 得到这个张量对应的python数值
#     loss = (y_pred - y).pow(2).sum()
#     print(t, loss.item())
#
#     # 使用autograd计算反向传播。这个调用将计算loss对所有requires_grad=True的tensor的梯度。
#     # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
#     loss.backward()
#     with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad
#
#         # 反向传播后手动将梯度设置为零
#         w1.grad.zero_()
#         w2.grad.zero_()

# 这个例子中，我们自定义一个自动求导函数来展示ReLU的非线性。并用它实现我们的两层网络：

class MyReLU(torch.autograd.Function):
    """
       我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，
       并完成张量的正向和反向传播。
       """

    @staticmethod
    def forward(ctx, x):
        """
        在正向传播中，我们接收到一个上下文对象和一个包含输入的张量；
        我们必须返回一个包含输出的张量，
        并且我们可以使用上下文对象来缓存对象，以便在反向传播中使用。
        """
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传播中，我们接收到上下文对象和一个张量，
        其包含了相对于正向传播过程中产生的输出的损失的梯度。
        我们可以从上下文对象中检索缓存的数据，
        并且必须计算并返回与正向传播的输入相关的损失的梯度。
        """
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# N是批大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# 产生随机权重的张量
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 正向传播：使用张量上的操作来计算输出值y；
    # 我们通过调用 MyReLU.apply 函数来使用自定义的ReLU
    y_pred = MyReLU.apply(x.mm(w1)).mm(w2)

    # 计算并输出loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用autograd计算反向传播过程。
    loss.backward()

    with torch.no_grad():
        # 用梯度下降更新权重
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 在反向传播之后手动清零梯度
        w1.grad.zero_()
        w2.grad.zero_()