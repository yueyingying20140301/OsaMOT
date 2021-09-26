import os
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import os.path
from PIL import Image
import PIL.ImageOps
import glob
import cv2
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
import shutil

print(torch.__version__)  #0.4.1
print(torchvision.__version__)  #0.2.1

#定义一些超参
train_batch_size = 32     #训练时batch_size
train_number_epochs = 25     #训练的epoch
modelnum = 0
detectionpath = "./detections/"
modelspath = "./Objectmodels/"
locationpath = "./location/"
trainpath = "./train_online/"
inputpath= "./input/"
savepath1= "./save/my_siamese1.pth"

def file(path):
    path=path
    list = []        # 空列表
# 遍历文件夹
    for root, dirs, files in os.walk(path):
# root 表示当前正在访问的文件夹路径;dirs 表示该文件夹下的子目录名list;files 表示该文件夹下的文件list
# 遍历文件
    #for f in files:
        #print(os.path.join(root, f))
# 遍历所有的文件夹
        for d in dirs:
            list.append(os.path.join(root, d))
    return list


def imshow(img,text=None,should_save=False):
    #展示一幅tensor图像，输入是(C,H,W)
    npimg = img.numpy() #将tensor转为ndarray
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #转换为(H,W,C)
    plt.show()

def imshow1(img1,img2):
    #展示2幅tensor图像，输入是(C,H,W)
    npimg1 = np.transpose(img1.numpy(),(1, 2, 0)) #将tensor转为ndarray,转换为(H,W,C)
    npimg2 = np.transpose(img2.numpy(),(1, 2, 0))  # 将tensor转为ndarray
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(npimg1)
    plt.subplot(1, 2, 2)
    plt.imshow(npimg2)
    plt.show()

def show_plot(iteration,loss):
    #绘制损失变化图
    plt.plot(iteration,loss)
    plt.show()

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def duplicates(num_list): #判断列表里是否有重复？
  if len(num_list)!=len(set(num_list)):
    return 1
  else:
    return 0

# 自定义Dataset类，__getitem__(self,index)每次返回(img1, img2, 0/1)
class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0, 1)  # 保证同类样本约占一半
        # random.randint(a,b)用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b。
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        #img0 = img0.convert("L") #img.convert('L')为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
        #img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0) #PIL.ImageOps.invert实现二值图像黑白反转
            img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
        #torch.from_numpy()方法把数组转换成张量,且二者共享内存
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# 搭建模型
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),  #nn.Linear（）是用于设置网络中的全连接层的
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    '''def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2'''

#自定义ContrastiveLoss siamese.py:150
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True) #求欧式距离
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def train1():
    # 定义文件dataset
    training_dir1 = "./data1/faces/training2/"   # 训练集地址
    folder_dataset = torchvision.datasets.ImageFolder(root=training_dir1)

    # 定义图像dataset
    transform1 = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.ToTensor()])
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transform1,
                                            should_invert=False)

    # 定义图像dataloader
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  batch_size=train_batch_size)

    # 可视化数据集
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                batch_size=8)
    example_batch = next(iter(vis_dataloader))  # 生成一批图像

    # 其中example_batch[0] 维度为torch.Size([8, 1, 100, 100])
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # print(concatenated.shape)
    # imshow(torchvision.utils.make_grid(concatenated, nrow=8))
    # print(example_batch[2].numpy())
    flag =0
    net = SiameseNetwork().cuda()  # 定义模型且移至GPU
    if flag:
        #训练模型
        criterion = ContrastiveLoss()  # 定义损失函数
        optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 定义优化器

        counter = []
        loss_history = []
        iteration_number = 0

        # 开始训练
        for epoch in range(0, train_number_epochs):
            for i, data in enumerate(train_dataloader, 0):
                img0, img1, label = data
                # img0维度为torch.Size([32, 3, 100, 100])，32是batch，label为torch.Size([32, 1])
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()  # 数据移至GPU
                optimizer.zero_grad()
                output1 = net(img0)
                output2 = net(img1)
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                optimizer.step()
                if i % 10 == 0:
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
            print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))
        show_plot(counter, loss_history)
        torch.save(net.state_dict(), savepath1)
    else:
        net.load_state_dict(torch.load(savepath1))
        net.eval()
    return net
def compare1(img1,img2):
    transform1 = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.ToTensor()])
    image1 = Image.open(img1)
    image1_tensor = transform1(image1)
    image2 = Image.open(img2)
    image2_tensor = transform1(image2)
    x0 = torch.ones([1, 3, 100, 100])
    x1 = torch.ones([1, 3, 100, 100])
    x0[0] = image1_tensor
    x1[0] = image2_tensor
    output1 = net1(x0.cuda())
    output2 = net1(x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    return euclidean_distance

net1 = train1()
t1 = compare1("11.jpg","person-1.jpg")
t2 = compare1("11.jpg","person-2.jpg")
t3 = compare1("11.jpg","person-3.jpg")
t4 = compare1("11.jpg","person-4.jpg")
t5 = compare1("11.jpg","person-5.jpg")
t6 = compare1("11.jpg","person-6.jpg")
t7 = compare1("11.jpg","person-7.jpg")
t8 = compare1("11.jpg","person-8.jpg")

print(t1,t2,t3,t4,t5,t6,t7,t8)

