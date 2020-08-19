from __future__ import print_function, division

'''PyTorch之数据加载和处理
PyTorch提供了许多工具来简化和希望数据加载，使代码更具可读性。'''

'''scikit-image：用于图像的IO和变换
pandas：用于更容易地进行csv解析'''


import os
import torch
import  pandas as pd  #用于更容易地进行csv解析
from skimage import io,transform  #用于图像的IO和变换
import  numpy as np
import  matplotlib.pyplot as plt
from torch.utils.data import  DataLoader, Dataset
from torchvision import transforms,utils


# 忽略警告
import warnings
warnings.filterwarnings("ignore")
plt.ion()  # interactive mode

# 读取数据集
'''将csv中的标注点数据读入（N，2）数组中，其中N是特征点的数量。读取数据代码如下：'''
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv') # 直接读取csv的方法.

n = 68
img_name = landmarks_frame.iloc[n,0]
landmarks = landmarks_frame.iloc[n,1:].as_matrix() # 是转换为矩阵吗
landmarks = landmarks.astype('float').reshape(-1,2)

print('Image name:{}'.format(img_name))
print('Landmarks shape:{}'.format(landmarks.shape))
print('First 4 Landmarks:{}'.format(landmarks[:4]))

'''4 编写函数
写一个简单的函数来展示一张图片和它对应的标注点作为例子。'''
def show_landmarks(image,landmarks):
    '''显示带有地标的图片'''
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,marker='.',c='r')
    plt.pause(0.01)
plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/',img_name)),landmarks)

plt.show()


# 数据集类
'''torch.utils.data.Dataset是表示数据集的抽象类，
因此自定义数据集应继承Dataset并覆盖以下方法 * __len__ 实现 len(dataset) 返还数据集的尺寸。 
* __getitem__用来获取一些索引数据，例如 dataset[i] 中的(i)。'''

'''5.1 建立数据集类
为面部数据集创建一个数据集类。
我们将在 __init__中读取csv的文件内容，
在 __getitem__中读取图片。这么做是为了节省内存 空间。
只有在需要用到图片的时候才读取它而不是一开始就把图片全部存进内存里。'''

'''我们的数据样本将按这样一个字典{'image': image, 'landmarks': landmarks}组织。 
我们的数据集类将添加一个可选参数transform 以方便对样本进行预处理。
下一节我们会看到什么时候需要用到transform参数。 __init__方法如下图所示：'''

class FaceLandmarksDateset(Dataset):
    '''面部标记数据'''
    def __init__(self,csv_file,root_dir,transform = None):
        '''
        :param csv_file:  带有注释的csv文件的路径
        :param root_dir: 包含所有图像的目录.
        :param transform: 一个样本上的可用的可选变换
        '''
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)  # 有多少个数据
    def __getitem__(self, item): # 默认指定
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[item,0]) # 名字是一个路径
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[item,1:]
        landmarks = np.array([landmarks]) #注意这个地方加了方括号的
        landmarks = landmarks.astype('float').reshape(-1,2) # 通过astype转变数据类型
        sample = {'image':image,'landmarks':landmarks}
        if self.transform:
            sample = self.transform(sample) # 这个其他的一个函数
        return sample

'''6.数据可视化
实例化这个类并遍历数据样本。我们将会打印出前四个例子的尺寸并展示标注的特征点。 代码如下图所示：'''
face_dataset = FaceLandmarksDateset(csv_file='data/faces/face_landmarks.csv',root_dir='data/faces/')
fig = plt.figure()
for i in range(len(face_dataset)): # 调用了face_dataset的__len__函数
    sample = face_dataset[i]
    print(i,sample['image'].shape,sample['landmarks'].shape)
    ax = plt.subplot(1,4,i+1) # 前四个
    plt.tight_layout()  # 这个函数是什么意思
    ax.set_title("Sample #{}".format(i))
    ax.axis('off')  # 这个off什么意思吗
    show_landmarks(**sample) # 这个**sample又是什么意思呢

    if i==3:
        plt.show()
        break
    # 为什么我输出的是分开的4张图,而作者的是一张图呢.

'''
7.数据变换
通过上面的例子我们会发现图片并不是同样的尺寸。
绝大多数神经网络都假定图片的尺寸相同。\
因此我们需要做一些预处理。让我们创建三个转换: 
* Rescale：缩放图片 
* RandomCrop：对图片进行随机裁剪。这是一种数据增强操作 
* ToTensor：把numpy格式图片转为torch格式图片 (我们需要交换坐标轴).
'''

'''我们会把它们写成可调用的类的形式而不是简单的函数，
这样就不需要每次调用时传递一遍参数。我们只需要实现__call__方法，必 要的时候实现 __init__方法。
我们可以这样调用这些转换:'''

# tsfm = Transform(params)
# transformed_sampe = tsfm(sample)

'''观察下面这些转换是如何应用在图像和标签上的。'''

class Rescale(object):
    """将样本中的图像重新缩放到给定大小。.

        Args:
            output_size（tuple或int）：所需的输出大小。 如果是元组，则输出为
             与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,smaple):
        image,landmarks = sample['image'],sample['landmarks']
        h,w = image.shape[:2]  #取前两个,第三个是通道
        if isinstance(self.output_size,int):  # 输入为int的情况
            if h>w:
                new_h,new_w = self.output_size*h/w,self.output_size
            else:
                new_h,new_w = self.output_size,self.output_size*w/h
        else:
            new_h,new_w = self.output_size
        new_h,new_w = int(new_h),int(new_w)
        img = transform.resize(image,(new_h,new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w/w,new_h/h] #对landmarks的坐标进行放缩.
        return {'image':img,'landmarks':landmarks}

class Rescale(object):
    pass


