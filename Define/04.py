'''
通常来说，当你处理图像，文本，语音或者视频数据时，你可以使用标准 python 包将数据加载成 numpy 数组格式，然后将这个数组转换成 torch.*Tensor
对于图像，可以用 Pillow，OpenCV
对于语音，可以用 scipy，librosa
对于文本，可以直接用 Python 或 Cython 基础数据加载模块，或者用 NLTK 和 SpaCy
'''

'''
特别是对于视觉，我们已经创建了一个叫做 totchvision 的包，
该包含有支持加载类似Imagenet，CIFAR10，MNIST 等公共数据集
的数据加载模块 torchvision.datasets 和支持加载图像数据数据转换模块 torch.utils.data.DataLoader。
'''


'''
对于本教程，我们将使用CIFAR10数据集，
它包含十个类别：‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’。
CIFAR-10 中的图像尺寸为33232，也就是RGB的3层颜色通道，每层通道内的尺寸为32*32。
'''

'''
训练一个图像分类器
我们将按次序的做如下几步：
1. 使用torchvision加载并且归一化CIFAR10的训练和测试数据集
2. 定义一个卷积神经网络
3. 定义一个损失函数
4. 在训练样本数据上训练网络
5. 在测试样本数据上测试网络
'''

'''加载并归一化 CIFAR10 使用 torchvision ,用它来加载 CIFAR10 数据非常简单。'''
import torch
import torchvision
import  torchvision.transforms as transforms

'''torchvision 数据集的输出是范围在[0,1]之间的 PILImage，我们将他们转换成归一化范围为[-1,1]之间的张量 Tensors。'''

transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)


trainset = torchvision.datasets.CIFAR10(root='./data',train = True,download=True,transform=transform)  # 数据集会自动下载.

trainloader = torch.utils.data.DataLoader(trainset,batch_size = 4,shuffle = True,num_workers = 2)  # 注意这个batch_size

testset = torchvision.datasets.CIFAR10(root='./data',train = False, download=True,transform= transform)

testloader = torch.utils.data.DataLoader(testset,batch_size = 4, shuffle = False, num_workers = 2) # shuffle 是什么含义呢

classes= ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 这是一个元组.


# 让我们来展示其中的一些训练图片。

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img /2 +0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))  #这些函数是什么意思都不清楚.
    plt.show()

dataiter = iter(trainloader)  #了解一下迭代器这个函数
images,labels = dataiter.next() # 我怎么知道这个迭代器返回的数据格式

# show image
imshow(torchvision.utils.make_grid(images)) #make_grid函数???
print(''.join('%5s'% classes[labels[j]] for j in range(4))) #为什么是4呢? batch_size = 4

'''定义一个卷积神经网络 在这之前先 从神经网络章节 复制神经网络，并修改它为3通道的图片(在此之前它被定义为1通道)'''
import  torch.nn as nn
import  torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2) # 卷积核大小吗?
        self.conv2 = nn.Conv2d(6,16,5) # 最后一位是卷积核大小
        self.fc1 = nn.Linear(16 * 5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x): # 前向计算的过程
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
'''定义一个损失函数和优化器 让我们使用分类交叉熵Cross-Entropy 作损失函数，动量SGD做优化器。'''

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.001, momentum = 0.9)  #决定了反向传播的过程

'''训练网络 这里事情开始变得有趣，
我们只需要在数据迭代器上循环传给网络和优化器 输入就可以。'''

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inputs,labels = data
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        running_loss +=loss.item()
        if i%2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

'''好的，第一步，让我们从测试集中显示一张图像来熟悉它。'''
outputs = net(images)
'''输出是预测与十个类的近似程度，与某一个类的近似程度越高，
网络就越认为图像是属于这一类别。所以让我们打印其中最相似类别类标：'''
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
