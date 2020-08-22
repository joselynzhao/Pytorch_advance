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


import torch
dtype = torch.float
# device = torch.device('cpu')
device = torch.device('cuda') #取消注释以在GPU上运行
# N是批量大小; D_in是输入维度;
# H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10
#创建随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 前向传递：计算预测y
    h = x.mm(w1)  # mm表示点成
    h_relu = h.clamp(min=0) # relu函数
    y_pred = h_relu.mm(w2)

    # 计算和打印损失
    loss = (y_pred - y).pow(2).sum().item() #．item是什么意思呢?
    print(t, loss)

    # Backprop计算w1和w2相对于损耗的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()  #对应copy
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    # 测试了速度,确实快很多.