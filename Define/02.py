# pytorch_自动微分.

'''autograd 自动微分. '''

import torch
# import torch.Tensor  #核心类 .requires_grad = True,则跟踪tensor的了所有操作.
# 完成计算后，您可以调用 .backward() 来自动计算所有梯度。
# 该张量的梯度将累积到 .grad 属性中。
# 要停止 tensor 历史记录的跟踪，您可以调用 .detach()，它将其与计算历史记录分离，并防止将来的计算被跟踪。

'''
要停止跟踪历史记录（和使用内存），您还可以将代码块使用 with torch.no_grad(): 包装起来。
在评估模型时，这是特别有用，因为模型在训练阶段具有 requires_grad = True 的可训练参数有利于调参，但在评估阶段我们不需要梯度。

还有一个类对于 autograd 实现非常重要那就是 Function。
Tensor 和 Function 互相连接并构建一个非循环图，它保存整个完整的计算过程的历史信息。
每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用，（如果用户自己创建张量，则g rad_fn 是 None ）。

如果你想计算导数，你可以调用 Tensor.backward()。
如果 Tensor 是标量（即它包含一个元素数据），则不需要指定任何参数backward()，
但是如果它有更多元素，则需要指定一个gradient 参数来指定张量的形状。
'''

x = torch.ones(2,2,requires_grad=True) # 跟踪计算
print(x)
y = x+2
print(y) #y 作为操作的结果被创建，所以它有 grad_fn
print(y.grad_fn)

z = y*y*3
out = z.mean()  #求了所有元素的平均值.
print(z,out)

a = torch.rand(2,2)
print('a-',a)
a = torch.randn(2,2)
print('a',a)   #感觉a 和a-没有什么区别.
a = ((a*3)/(a-1))
print(a.requires_grad)

a.requires_grad_(True) #修改这个值.
print(a.requires_grad)
print('a',a.grad_fn)  # 自己创建的是没有的.

b = (a*a).sum()
print(b.requires_grad)
print(b.grad_fn)

out.backward() #等同于out.backward(torch.tensor(1.))。
print(x)
print(x.grad) #要进行后向传播之后才会有梯度.


# 现在让我们看一个雅可比向量积的例子：
x = torch.randn(3,requires_grad=True)  #如果requires_grad = False, 那y就没有了grad_fn.
print(x) #tensor([-0.7594,  0.7919,  0.8072], requires_grad=True)
y = x*2
print(y.data.norm()) #tensor(6.7113)
while y.data.norm() <1000:
    y = y *2
print(y) #tensor([-148.7198, 1301.4535, 1111.7496], grad_fn=<MulBackward0>)

'''现在在这种情况下，y 不再是一个标量。
torch.autograd 不能够直接计算整个雅可比，但是如果我们只想要雅可比向量积，
只需要简单的传递向量给 backward 作为参数。'''

v = torch.tensor([0.1,1.0,0.00001],dtype=torch.float)
y.backward(v)
print(x.grad) # 不明白为什么影响的是x的梯度. tensor([5.1200e+01, 5.1200e+02, 5.1200e-03])


#你可以通过将代码包裹在 with torch.no_grad()，来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导。
print(x.requires_grad) # True
print((x ** 2).requires_grad) # True
with torch.no_grad():
    print((x ** 2).requires_grad) # False


