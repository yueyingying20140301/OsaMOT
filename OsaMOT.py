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
import random
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import shutil
from tracking2 import *
from train1 import *
import colorsys
import random

#定义一些超参
train_batch_size = 32     #训练时batch_size
train_number_epochs = 50   #训练的epoch
modelnum = 0
detectionpath = "./detections/"
modelspath = "./Objectmodels/"
modelspath1 = "./Objectmodels1/"
locationpath = "./location/"
trainpath = "./train_online/"
inputpath= "./input/"
savepath= "./save/my_siamese.pth"
result1path = "./result/MOT1.txt"
result2path = "./result/MOT2.txt"
numClass=5

def compare(img1,img2):
    image1 = Image.open(img1)
    image1_tensor = transform(image1)
    image2 = Image.open(img2)
    image2_tensor = transform(image2)
    x0 = torch.ones([1, 3, 200, 200])
    x1 = torch.ones([1, 3, 200, 200])
    x0[0] = image1_tensor
    x1[0] = image2_tensor
    output1 = net(x0.cuda())
    output2 = net(x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    return euclidean_distance

def computefeature(img1):
    image1 = Image.open(img1)
    image1_tensor = transform(image1)
    x0 = torch.ones([1, 3, 200, 200])
    x0[0] = image1_tensor
    output = net(x0.cuda())
    return output

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

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors

def model_clean(modelspath):  #删除模板库中相似的目标
    models = sorted(glob.glob(os.path.join(modelspath + '*.jpg')))  # 读取目标模板
    num = len(models)
    distance = np.zeros([num, num])
    for i in range(num-1):
        for j in range(i+1,num):
            distance[i][j] = compare(models[i],models[j])
    for i in range(num - 1):
        for j in range(i + 1, num):
            if(distance[i][j]<0.3):
                os.remove(models[j])
                print("deleted")
                print(j)

def occlusion_judge(image,dboxs):
    image1=image
    boxnum = dboxs.shape[0]
    size = image.shape
    for i in range(size[0]):
        for j in range(size[1]):
            ppp=0
            for k in range(boxnum):
                if((j>=dboxs[k][0][0] and j<=dboxs[k][0][2]) and (i>=dboxs[k][0][1] and i<=dboxs[k][0][3])):
                    ppp=ppp+1
            if(ppp==0):
                image1[i][j] = [255,255,255] #白色
            if(ppp==1):
                image1[i][j] = [0,255,255] #黄色
            if(ppp==2):
                image1[i][j] = [255, 0, 255] #粉红
            if(ppp>=3):
                image1[i][j] = [0, 0, 255] #红色
    return image1

def occlusion_judge1(dboxs): #全局遮挡系数计算
    boxnum = dboxs.shape[0]
    occlusionnum=0
    for k in range(boxnum-1):
        for g in range(k+1,boxnum):
            rec1 = int(dboxs[k][0][0]), int(dboxs[k][0][1]), int(dboxs[k][0][2]), int(dboxs[k][0][3])
            rec2 = int(dboxs[g][0][0]), int(dboxs[g][0][1]), int(dboxs[g][0][2]), int(dboxs[g][0][3])
            az=compute_iou(rec1, rec2)
            if(az!=0):
                occlusionnum=occlusionnum+1
    rate=occlusionnum/boxnum
    return rate

def occlusion_judge2(dboxs):  #局部遮挡系数计算
    boxnum = dboxs.shape[0]
    occlusionmatri = [0 for x in range(0,50)] #记录当前帧每个目标的遮挡系数
    for k in range(boxnum):
        mianji = abs((dboxs[k][0][2]-dboxs[k][0][0])*(dboxs[k][0][3]-dboxs[k][0][1]))
        az = 0
        for g in range(boxnum):
            if(k!=g):
                rec1 = int(dboxs[k][0][0]), int(dboxs[k][0][1]), int(dboxs[k][0][2]), int(dboxs[k][0][3])
                rec2 = int(dboxs[g][0][0]), int(dboxs[g][0][1]), int(dboxs[g][0][2]), int(dboxs[g][0][3])
                az = az + compute_intersect(rec1, rec2)
        occlusionmatri[k] = az/mianji
    return occlusionmatri

#定义文件dataset
#training_dir = "./data1/faces/training2/"  #训练集地址
#training_dir = "/media/yyy/9B5BAA2D94B22124/Dataset/prid_2011/multi_shot/cam_b/"  #训练集地址
training_dir = "/media/yyy/9B5BAA2D94B22124/Dataset/i-LIDS-VID/sequences/cam1/"
folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)

#定义图像dataset
transform = transforms.Compose([transforms.Resize((200,200)),
                                transforms.ToTensor()])
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False)

#定义图像dataloader
train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=train_batch_size)

# 可视化数据集
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        batch_size=8)
example_batch = next(iter(vis_dataloader)) #生成一批图像

#其中example_batch[0] 维度为torch.Size([8, 1, 100, 100])
#concatenated = torch.cat((example_batch[0],example_batch[1]),0)
#print(concatenated.shape)
#imshow(torchvision.utils.make_grid(concatenated, nrow=8))
#print(example_batch[2].numpy())



def train():
    flag1 = 0#1:需要训练 0:不训练，直接加载模型
    res50 = models.resnet50(pretrained=True)
    numFit = res50.fc.in_features
    res50.fc = nn.Linear(numFit, numClass)
    net = res50.cuda()
    if flag1:
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
        torch.save(net.state_dict(), savepath)

    else:
        net.load_state_dict(torch.load(savepath))
        net.eval()

    return net

colors = ncolors(600)
random.shuffle(colors)
#print(colors)


with torch.no_grad():
    net = train()
    net1 = train1()
    list1 = sorted(file(detectionpath)) #每一帧检测到的目标的文件夹名称构成列表
    frames = len(list1)
    locations = sorted(glob.glob(os.path.join(locationpath + '*.txt'))) #按帧划分，每一帧中目标的坐标位置
    inputs = sorted(glob.glob(os.path.join(inputpath + '*.jpg'))) #输入图像序列
    toc = 0
    od = torch.zeros([600, 2000, 1, 4]) #轨迹tensor，目标数300，帧数1000，坐标值为1行4列
    feature = torch.zeros([600, 2000, 1, 5]) #所有目标历史特征，目标数300，帧数1000，特征和值为1行5列
    feature_flag = torch.zeros([600, 2000, 1])  # 所有目标历史关联标记，目标数300，帧数1000，1：关联；0：未关联
    feature_weight = torch.zeros([600, 2000, 1])  # 所有目标历史特征权重，由局部遮挡系数得到，目标数300，帧数1000，值为0到1之间的数
    model_feature = torch.zeros([600, 1, 5]) #模板历史特征，目标数300，特征值为1行5列
    current_feature = torch.zeros([50, 1, 5]) #当前帧目标特征，目标数30，特征值为1行5列
    I = [-1 for x in range(0,600)] #记录每个目标进入场景的开始帧 如I【1】=5代表1号目标在第5帧进入场景
    om = [0 for x in range(0,600)] #记录每个目标连续匹配的次数
    om1 = [0 for x in range(0,600)] #记录每个目标连续未匹配的次数
    occlusionmatri = [0 for x in range(0, 100)]  # 记录当前帧每个目标的遮挡系数


    for f in range(frames):
        print("第%d帧" % f)
        if (f<=frames/2):
            fp = open(result1path,'a')
        else:
            fp = open(result2path, 'a')
        tic = cv2.getTickCount()  # 记录当前时间
        path1 = list1[f]+'/'  #图像读取地址
        path2 = list1[f]+'/'  #图像读取地址
        models = sorted(glob.glob(os.path.join(modelspath + '*.jpg')))  # 所有目标模板文件夹
        imgs1 = glob.glob(os.path.join(path1 + '*.jpg')) # 前一帧图片路径
        imgs2 = glob.glob(os.path.join(path1 + '*.jpg')) # 后一帧图片路径
        imgs1.sort(key=lambda x: int(x.split('person-')[1].split('.jpg')[0]))
        imgs2.sort(key=lambda x: int(x.split('person-')[1].split('.jpg')[0]))
        oimg1 = cv2.imread(inputs[f])
        a = np.loadtxt(locations[f]) #获取每一帧中检测到的目标的坐标
        b = np.reshape(a, (-1, 1, 4))  # 坐标框
        objectnum = b.shape[0]
        c = b[:, :, 0:2]  # 中心坐标

        a1 = np.loadtxt(locations[f])  # 获取每一帧中检测到的目标的坐标
        b1 = np.reshape(a1, (-1, 1, 4))  # 坐标框
        objectnum1 = b1.shape[0]

        oimg = cv2.imread(inputs[f])
        oimg2 = oimg1
        #给每个目标画框
        #for q in range(objectnum):
            #cv2.rectangle(oimg, (int(b[q][0][0]), int(b[q][0][1])), (int(b[q][0][2]), int(b[q][0][3])), (0, 255, 0), 3)
        num1 = len(models) #控制目标模板数量
        clean_flag=0
        '''if(num1>=299):
            for i in range(num1):
                os.remove(models[i])
                modelnum=0
                clean_flag=1'''
        num2 = len(imgs2)
        if(num2==0):
            continue
        print(num1,num2)
        one = np.ones([num1, num2])
        '''for i in range(num1):
            for j in range(num2):
                one[i][j] = one[i][j]+1'''
        s = np.zeros([num1,num2]) #高层特征相似度矩阵
        ss = np.zeros([num1,num2]) #历史特征相似度矩阵
        sss = np.zeros([num1,num2]) #直方图相似度矩阵
        judge = np.zeros([num1, num2])
        d = np.zeros([num1,num2]) #IOU矩阵
        nv = np.zeros([600,600])  #目标消失和新入矩阵，100个目标，100帧,匹配用1表示，不匹配用0表示
        history = [0 for x in range(0, 8)]  # 更新时使用，记录当前匹配目标与历史模板的相似度

        '''occlusion=occlusion_judge(oimg2,b1)
        cv2.imwrite('./occlusion/%d.jpg' % f, occlusion)'''

        rate = occlusion_judge1(b1) #计算全局遮挡系数
        occlusionmatri = occlusion_judge2(b1)  #计算局部遮挡系数
        print(rate)
        print(occlusionmatri)

        if (f==0 or clean_flag==1):     #第一帧时，将目标存入模板和轨迹矩阵
            i = 0
            for img1 in imgs1:
                modelnum = modelnum + 1
                new_obj_name = str(modelnum-1) + '.jpg'
                shutil.copy(img1, modelspath + new_obj_name)
                output = computefeature(img1)
                feature[i][f] = output
                feature_flag[i][f] = 1
                model_feature[i] = output
                od[i][f] = torch.from_numpy(b[i])
                I[modelnum-1]=0
                cv2.rectangle(oimg1, (int(b[i][0][0]), int(b[i][0][1])), (int(b[i][0][2]), int(b[i][0][3])),
                              (colors[i][0], colors[i][1], colors[i][2]), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(oimg1, str(i), (int(b[i][0][0]), int(b[i][0][1])), font, 1, (255, 0, 255), 2)
                fp.writelines(str(f+1) + ',' + str(modelnum) + ',' + str(b[i][0][0]) + ',' + str(b[i][0][1]) + ','+ str(b[i][0][2]-b[i][0][0]) + ',' + str(b[i][0][3]-b[i][0][1]) + ',-1,-1,-1,-1' + '\n')
                i = i+1
        else:
            '''if (f>=300 and f % 20 == 0):  # 每20帧在线训练一次低层特征提取网络
                print("train_online")
                net1 = train1()'''
            j = 0
            for img2 in imgs2:
                j = j + 1
                image2 = Image.open(img2)
                image2_tensor = transform(image2)
                x1 = torch.ones([1, 3, 200, 200])
                x1[0] = image2_tensor
                output2 = net(x1.cuda())
                current_feature[j - 1] = output2

            i = 0
            for img1 in models:
                i = i+1
                j = 0
                for img2 in imgs2:
                    j = j + 1
                    if(om1[i] >= 100): #连续100帧未匹配的不再考虑
                        ss[i - 1][j - 1] = 100
                    else:
                        history_distance = 0
                        matchnum = 1
                        for p in range(f):
                            if(feature_flag[i-1][p]==1):
                                output1 = feature[i-1][p].cuda()
                                output2 = current_feature[j - 1].cuda()
                                feature_weight=feature_weight.cuda()
                                dis1 = F.pairwise_distance(output1, output2)
                                if(feature_weight[i - 1][p] < 0.6):
                                    matchnum = matchnum + 1
                                    history_distance = history_distance + dis1
                                #history_distance = history_distance + dis1
                                #history_distance = history_distance + dis1*(feature_weight[i-1][p])
                        ss[i - 1][j - 1] = history_distance/matchnum
                        rect1 = (int(od[i - 1][f - 1][0][0]), int(od[i - 1][f - 1][0][1]), int(od[i - 1][f - 1][0][2]),
                             int(od[i - 1][f - 1][0][3]))
                        rect2 = (int(b1[j - 1][0][0]), int(b1[j - 1][0][1]), int(b1[j - 1][0][2]), int(b1[j - 1][0][3]))
                        d[i - 1][j - 1] = compute_iou(rect1, rect2)

            #print(d)
            d = np.subtract(one,d)
            for i in range(num1):
                for j in range(num2):
                    '''if(occlusionmatri[j]>0.6):
                        #judge[i][j] = rate / 2 * ss[i][j]
                        #judge[i][j] = rate/2*ss[i][j] + (1 - rate/2) * d[i][j] + sss[i][j]
                        judge[i][j] = (rate + occlusionmatri[j]) / 2 * ss[i][j] + (1 - rate / 2) * d[i][j]
                    else:'''
                    #judge[i][j] = rate / 2 * ss[i][j] + (1 - rate / 2) * d[i][j]
                    judge[i][j] = rate / 2 * ss[i][j] + (1 - rate / 2) * d[i][j]
            #judge = np.add(d,ss)
            #judge = np.add(s,d)
            #judge = np.add(judge,ss)
            #print(ss)
            #print(s)
            #print("****************************************")
            #print(d)
            #print("****************************************")
            #print(judge)
            s1 = np.array(judge)
            d = np.array(d)
            if (num1>num2):
                min_index0 = np.argmin(s1, axis=0)  # 每列最小值的行标
                for k in range(3):  # 如果有几对重复的话，需要多次扫描
                    if (duplicates(min_index0)):  #如果有重复,根据相似度特征值，保留值小的
                        for i in range(len(min_index0)-1):
                            for j in range(i+1,len(min_index0)):
                                if (min_index0[i]==min_index0[j]):
                                    if (d[min_index0[i]][i] != 1) and (d[min_index0[i]][j] != 1):
                                        if (judge[min_index0[i]][i] <= judge[min_index0[i]][j]):
                                            min_index0[j] = -1
                                        else:
                                            min_index0[i] = -1

                                        '''if (judge[min_index0[i]][i]>judge[min_index0[i]][j]):
                                            a = d[:,i]
                                            b = a.tolist()
                                            min_index0[i] = b.index(min(b))
                                        else:
                                            a = d[:,j]
                                            b = a.tolist()
                                            min_index0[j] = b.index(min(b))'''
                                    elif(d[min_index0[i]][i]==1): #如果没有IOU重叠，取消关联，用-1表示
                                        min_index0[i] = -1
                                    elif(d[min_index0[i]][j]==1):
                                        min_index0[j] = -1

                #print("当前帧目标-模板")
                print(min_index0)
                for m in range(num2):
                    if(min_index0[m]!=-1):
                        shutil.copy(imgs2[m],trainpath + str(min_index0[m]) + '/' + str(f + 1) + '.jpg')  # 把匹配的目标按目标号加入训练数据文件夹
                        feature[min_index0[m]][f] = current_feature[m]
                        feature_flag[min_index0[m]][f] = 1
                        feature_weight[min_index0[m]][f] = occlusionmatri[m]
                        if (ss[min_index0[m]][m] > 0.5):
                            '''if(f>7):
                                update = sorted(glob.glob(os.path.join(trainpath + str(min_index0[m]) + '/' +'*.jpg')))  # 输入图像序列
                                cn=len(update)
                                if(cn>8):
                                    cn=8
                                affinity=0
                                for i in range(cn):
                                    history[i] = compare1(imgs2[m],update[i])
                                    affinity=affinity+history[i]
                                affinity=affinity/cn
                                #print(m,cn,affinity)
                                if(affinity<1):
                                    obj_name = str(min_index0[m]) + '.jpg'
                                    shutil.copy(imgs2[m], modelspath + obj_name)
                                    model_feature[min_index0[m]] = current_feature[m]
                            else:'''
                            obj_name = str(min_index0[m]) + '.jpg'
                            shutil.copy(imgs2[m], modelspath + obj_name)
                            model_feature[min_index0[m]] = current_feature[m]
                        #if (judge[min_index0[m]][m] < 3):  # 更新模板
                        #if(f%10==0):
                        #if (s[min_index0[m]][m] > 0.5):  # 更新模板
                        '''if (s[min_index0[m]][m] > 0.5 and om[min_index0[m]] % 5 == 0):  # 更新模板
                            obj_name = str(min_index0[m]) + '.jpg'
                            shutil.copy(imgs2[m], modelspath + obj_name)'''
                        od[min_index0[m]][f] = torch.from_numpy(b1[m])  # 添加目标轨迹
                        # 给每个目标画框并添加目标ID
                        cv2.rectangle(oimg1, (int(b1[m][0][0]), int(b1[m][0][1])), (int(b1[m][0][2]), int(b1[m][0][3])),
                                      (colors[min_index0[m]][0], colors[min_index0[m]][1], colors[min_index0[m]][2]), 3)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(oimg1, str(min_index0[m]), (int(b1[m][0][0]), int(b1[m][0][1])), font, 1, (255, 0, 255),2)
                        om[min_index0[m]] = om[min_index0[m]]+1
                        fp.writelines(
                            str(f+1) + ',' + str(min_index0[m]+1) + ',' + str(b1[m][0][0]) + ',' + str(b1[m][0][1]) + ',' + str(
                                b1[m][0][2] - b1[m][0][0]) + ',' + str(b1[m][0][3] - b1[m][0][1]) + ',-1,-1,-1,-1' + '\n')
                    if (min_index0[m] == -1):
                        zz=0
                        for img3 in models:
                            distance = compare(imgs2[m],img3)
                            #print(m,distance)
                            if(distance<0.5):
                                zz=1
                                break
                        if (zz==0):
                            modelnum = modelnum + 1  # 未匹配的目标为新进入场景的目标（birth目标），加入目标模板集
                            shutil.copy(imgs2[m], trainpath + str(modelnum-1) + '/' + str(f + 1) + '.jpg')
                            model_feature[modelnum-1] = current_feature[m]
                            feature[modelnum-1][f] = current_feature[m]
                            feature_flag[modelnum - 1][f] = 1
                            feature_weight[modelnum - 1][f] = occlusionmatri[m]
                            new_obj_name = str(modelnum - 1) + '.jpg'
                            shutil.copy(imgs2[m], modelspath + new_obj_name)
                            od[modelnum - 1][f] = torch.from_numpy(b1[m])
                            om[modelnum - 1] = om[modelnum - 1] + 1
                            I[modelnum - 1] = f
                            cv2.rectangle(oimg1, (int(b1[m][0][0]), int(b1[m][0][1])), (int(b1[m][0][2]), int(b1[m][0][3])),
                                                      (colors[modelnum - 1][0], colors[modelnum - 1][1], colors[modelnum - 1][2]), 3)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(oimg1, str(modelnum - 1), (int(b1[m][0][0]), int(b1[m][0][1])), font, 1, (255, 0, 255),
                                                    2)
                            fp.writelines(str(f + 1) + ',' + str(modelnum) + ',' + str(b1[m][0][0]) + ',' + str(
                                        b1[m][0][1]) + ',' + str(
                                        b1[m][0][2] - b1[m][0][0]) + ',' + str(
                                        b1[m][0][3] - b1[m][0][1]) + ',-1,-1,-1,-1' + '\n')


                for n in range(num1):  #未匹配的模板目标,在上一帧该目标周围根据预测运动量进行寻找
                    if (n not in min_index0):
                        s1 = Image.open(models[n])
                        s1_tensor = transform(s1)
                        L = od[n][f-1]    #预测位置
                        L = L.int()
                        #计算n号目标在前面帧中，在x和y轴上的平均位移，用以估计后续目标的运动量
                        a11 = (od[n][I[n]][0][2]-od[n][I[n]][0][0])/2
                        b11 = (od[n][I[n]][0][3]-od[n][I[n]][0][1])/2
                        a22 = (od[n][f-1][0][2]-od[n][f-1][0][0])/2
                        b22 = (od[n][f-1][0][3]-od[n][f-1][0][1])/2
                        if (f-1-I[n]==0):
                            xa=0
                            xb=0
                        else:
                            xa = int((a22-a11)/(f-1-I[n]))
                            xb = int((b22-b11)/(f-1-I[n]))
                        if (L[0][0]!=0 and L[0][1]!=0):
                            s2 = oimg[L[0][1]+xb:L[0][3]+xb,L[0][0]+xa:L[0][2]+xa]
                            cv2.imwrite("new.jpg", s2)
                            try:
                                s2 = Image.open("new.jpg")
                            except:
                                continue
                            s2_tensor = transform(s2)
                            x00 = torch.ones([1, 3, 200, 200])
                            x11 = torch.ones([1, 3, 200, 200])
                            x00[0] = s1_tensor
                            x11[0] = s2_tensor
                            output1 = net(x00.cuda())
                            output2 = net(x11.cuda())
                            distance = F.pairwise_distance(output1, output2)
                            print(distance)
                            if (distance<0.8):
                                L[0][0] = L[0][0] + xa
                                L[0][2] = L[0][2] + xa
                                L[0][1] = L[0][1] + xb
                                L[0][3] = L[0][3] + xb
                                od[n][f] = L
                                cv2.rectangle(oimg1, (L[0][0], L[0][1]),(L[0][2], L[0][3]),(colors[n][0], colors[n][1], colors[n][2]), 3)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(oimg1, str(n), (L[0][0], L[0][1]), font, 1,(255, 0, 255),2)
                                shutil.copy("new.jpg", trainpath + str(n) + '/' + str(f + 1) + '.jpg')
                                feature[n][f] = output2
                                feature_flag[n][f] = 1
                                feature_weight[n][f] = 1
                                om[n] = om[n] + 1
                                L = L.numpy()
                                fp.writelines(
                                    str(f + 1) + ',' + str(n+1) + ',' + str(L[0][0]) + ',' + str(
                                        L[0][1]) + ',' + str(L[0][2] - L[0][0]) + ',' + str(L[0][3] - L[0][1]) + ',-1,-1,-1,-1' + '\n')
                            else:
                                om[n] = 0

            if (num1<=num2):
                min_index1 = np.argmin(s1, axis=1)  # 每行最小值的列标
                for k in range(3):     #如果有几对重复的话，需要多次扫描
                    if (duplicates(min_index1)):  #如果有重复,根据相似度特征值，保留值小的
                        for i in range(len(min_index1)-1):
                            for j in range(i+1,len(min_index1)):
                                if (min_index1[i]==min_index1[j]):
                                    if (d[i][min_index1[i]] != 1) and (d[j][min_index1[i]] != 1):
                                        if (judge[i][min_index1[i]] <= judge[j][min_index1[i]]):
                                            min_index1[j] = -1
                                        else:
                                            min_index1[i] = -1
                                        '''if (judge[i][min_index1[i]]>judge[j][min_index1[i]]):
                                            a = d[i,:]
                                            b = a.tolist()
                                            min_index1[i] = b.index(min(b))
                                        else:
                                            a = d[j,:]
                                            b = a.tolist()
                                            min_index1[j] = b.index(min(b))'''
                                    elif (d[i][min_index1[i]]==1): #如果没有IOU重叠，取消关联，用-1表示
                                        min_index1[i] = -1
                                    elif (d[j][min_index1[i]]==1):
                                        min_index1[j] = -1

                #print("模板-当前帧目标")
                print(min_index1)
                for n in range(num1):
                    if (min_index1[n]!=-1) :
                        shutil.copy(imgs2[min_index1[n]],trainpath + str(n) + '/' + str(f + 1) + '.jpg')  # 把匹配的目标按目标号加入训练数据文件夹'''
                        feature[n][f] = current_feature[min_index1[n]]
                        feature_flag[n][f] = 1
                        feature_weight[n][f] = occlusionmatri[min_index1[n]]
                        om[n] = om[n] + 1
                        '''if (s[n][min_index1[n]] < 0.5):
                            obj_name = str(n) + '.jpg'
                            shutil.copy(imgs2[min_index1[n]], modelspath + obj_name)'''
                        #if (judge[n][min_index1[n]] < 3):  # 更新模板
                        #if(f%10==0):
                        #if (s[n][min_index1[n]] > 0.5):  # 更新模板
                        if (ss[n][min_index1[n]] > 0.5):  #更新模板
                            obj_name = str(n) + '.jpg'
                            shutil.copy(imgs2[min_index1[n]], modelspath + obj_name)
                            model_feature[n] = current_feature[min_index1[n]]
                        pp = min_index1[n]
                        od[n][f] = torch.from_numpy(b1[pp])
                        print(colors[n])
                        cv2.rectangle(oimg1, (int(b1[min_index1[n]][0][0]), int(b1[min_index1[n]][0][1])), (int(b1[min_index1[n]][0][2]), int(b1[min_index1[n]][0][3])),
                                      (colors[n][0], colors[n][1], colors[n][2]), 3)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(oimg1, str(n), (int(b1[min_index1[n]][0][0]), int(b1[min_index1[n]][0][1])), font, 1, (255, 0, 255),2)
                        om[n] = om[n] + 1
                        fp.writelines(
                            str(f + 1) + ',' + str(n+1) + ',' + str(b1[min_index1[n]][0][0]) + ',' + str(
                                b1[min_index1[n]][0][0]) + ',' + str(
                                b1[min_index1[n]][0][2] - b1[min_index1[n]][0][0]) + ',' + str(b1[min_index1[n]][0][3] - b1[min_index1[n]][0][1]) + ',-1,-1,-1,-1' + '\n')
                    if (min_index1[n]==-1):
                        s1 = Image.open(models[n])
                        s1_tensor = transform(s1)
                        L = od[n][f - 1]  # 预测位置
                        L = L.int()
                        # 计算n号目标在前面帧中，在x和y轴上的平均位移，用以估计后续目标的运动量
                        a11 = (od[n][I[n]][0][2] - od[n][I[n]][0][0]) / 2
                        b11 = (od[n][I[n]][0][3] - od[n][I[n]][0][1]) / 2
                        a22 = (od[n][f - 1][0][2] - od[n][f - 1][0][0]) / 2
                        b22 = (od[n][f - 1][0][3] - od[n][f - 1][0][1]) / 2
                        if (f - 1 - I[n] == 0):
                            xa = 0
                            xb = 0
                        else:
                            xa = int((a22 - a11) / (f - 1 - I[n]))
                            xb = int((b22 - b11) / (f - 1 - I[n]))
                        if (L[0][0] != 0 and L[0][1] != 0):
                            s2 = oimg[L[0][1] + xb:L[0][3] + xb, L[0][0] + xa:L[0][2] + xa]
                            cv2.imwrite("new.jpg", s2)
                            try:
                                s2 = Image.open("new.jpg")
                            except:
                                continue
                            s2_tensor = transform(s2)
                            x00 = torch.ones([1, 3, 200, 200])
                            x11 = torch.ones([1, 3, 200, 200])
                            x00[0] = s1_tensor
                            x11[0] = s2_tensor
                            output1 = net(x00.cuda())
                            output2 = net(x11.cuda())
                            distance = F.pairwise_distance(output1, output2)
                            print(distance)
                            if (distance < 0.8):
                                L[0][0] = L[0][0] + xa
                                L[0][2] = L[0][2] + xa
                                L[0][1] = L[0][1] + xb
                                L[0][3] = L[0][3] + xb
                                od[n][f] = L
                                cv2.rectangle(oimg1, (L[0][0], L[0][1]), (L[0][2], L[0][3]), (colors[n][0], colors[n][1], colors[n][2]), 3)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(oimg1, str(n), (L[0][0], L[0][1]), font, 1, (255, 0, 255), 2)
                                shutil.copy("new.jpg", trainpath + str(n) + '/' + str(f + 1) + '.jpg')
                                feature[n][f] = output2
                                feature_flag[n][f] = 1
                                feature_weight[n][f] = 1
                                om[n] = om[n] + 1
                                L = L.numpy()
                                fp.writelines(
                                    str(f + 1) + ',' + str(n+1) + ',' + str(L[0][0]) + ',' + str(
                                        L[0][1]) + ',' + str(L[0][2] - L[0][0]) + ',' + str(L[0][3] - L[0][1]) + ',-1,-1,-1,-1' + '\n')
                            else:
                                om[n] = 0

                for m in range(num2):
                    if (m not in min_index1):
                         zz=0
                         for img3 in models:
                             distance = compare(imgs2[m],img3)
                             #print(m,distance)
                             if(distance<0.5):
                                 zz=1
                                 break
                         if (zz==0):
                             modelnum = modelnum + 1  # 未匹配的目标为新进入场景的目标（birth目标），加入目标模板集
                             shutil.copy(imgs2[m], trainpath + str(modelnum - 1) + '/' + str(f + 1) + '.jpg')
                             new_obj_name = str(modelnum-1) + '.jpg'
                             shutil.copy(imgs2[m], modelspath + new_obj_name)
                             model_feature[modelnum - 1] = current_feature[m]
                             feature[modelnum - 1][f] = current_feature[m]
                             feature_flag[modelnum - 1][f] = 1
                             feature_weight[modelnum - 1][f] = occlusionmatri[m]
                             od[modelnum-1][f] = torch.from_numpy(b1[m])
                             om[modelnum-1] = om[modelnum-1] + 1
                             I[modelnum-1]=f
                             cv2.rectangle(oimg1, (int(b1[m][0][0]), int(b1[m][0][1])),(int(b1[m][0][2]), int(b1[m][0][3])),(colors[modelnum-1][0], colors[modelnum-1][1], colors[modelnum-1][2]), 3)
                             font = cv2.FONT_HERSHEY_SIMPLEX
                             cv2.putText(oimg1, str(modelnum-1), (int(b1[m][0][0]), int(b1[m][0][1])), font, 1,(255, 0, 255),2)
                             fp.writelines(str(f + 1) + ',' + str(modelnum) + ',' + str(b1[m][0][0]) + ',' + str(
                                                b1[m][0][1]) + ',' + str(
                                                b1[m][0][2] - b1[m][0][0]) + ',' + str(
                                                b1[m][0][3] - b1[m][0][1]) + ',-1,-1,-1,-1' + '\n')
        #print(om)
        for i in range(num1): #统计未匹配次数
            if (om[i]==0):
                om1[i] = om1[i]+1
        print(om1)
        #print(feature_weight)
        '''for i in range(num1):   #打印轨迹
            if(om[i]!=0):
                for j in range(f):
                    cv2.circle(oimg1, ((od[i][j][0][0]+(od[i][j][0][2]-od[i][j][0][0])/2), (od[i][j][0][1]+(od[i][j][0][3]-od[i][j][0][1])/2)), 6, (colors[i][0], colors[i][1], colors[i][2]), -1)
                    #print(i,j,od[i][j])'''

        cv2.imshow('tracking result', oimg1)
        cv2.imwrite('./track_result/%d.jpg' % f, oimg1)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    #print(feature_weight)
    print('Time: {:02.1f}s Speed: {:3.1f} fps'.format(toc,fps))
    print(I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    fp.close()