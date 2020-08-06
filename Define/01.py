# from __future__ import print_function

import  torch

x = torch.empty(5,3) # 不进行初始化
print(x)  # 生成了5*3的张亮

x = torch.rand(5,3) # 随机初始化
print(x)

x = torch.zeros(5,3,dtype=torch.long) # all are zero,
print(x)

x = torch.tensor([5.5,3]) # 构造一个张量
print(x) #如果是浮点数,小数点是4位

x = x.new_ones(5,3,dtype=torch.double)
print(x)
x = torch.randn_like(x,dtype=torch.float)

print(x)
print(x.size())  #元组.
print(x.size()[0])


# 操作
y = torch.rand(5,3)
print(x,y)
print(x+y)
print(torch.add(x,y))  #可以直接加,用可以通过torch.add 加,那么有什么去区别呢?

result = torch.empty(5,3)
torch.add(x,y,out = result)   #这个result必须要先声明.
print(result)

#in_place
y.add_(x)
print(y)  #会修改y的值
print(x)
print(x[:,1])

# 想要改变一个tensor的大小
x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8)  # -1 表示由其他维度推断.
print(x.size(),y.size(),z.size())  # torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])


x = torch.randn(1)   # randn和rand的区别.
print(x)
print(x.item())