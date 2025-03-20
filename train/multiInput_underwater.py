import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_ssim
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.append('/data/sz3/temp')
from model.GIDC import GIDC, _initialize_weights
from model.ResUNet import ResUNet, _initialize_weights

seed = 99
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# define TV loss and loss
def tv_regularization(img):
    dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return torch.mean(dx) + torch.mean(dy)

def loss_fn(y, y_pred):
    return torch.mean(torch.square(y - y_pred))

def loss_ssim(image1,image2):
    return pytorch_ssim.ssim(image1,image2) 

def psnr(target, prediction):
    mse = F.mse_loss(target, prediction)  
    max_pixel = torch.max(target)  
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))  
    return psnr

def binarize_matrix(matrix, threshold):
    rows = len(matrix)
    cols = len(matrix[0])
    binary_matrix = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] >= threshold:
                binary_matrix[i][j] = 255
            else:
                binary_matrix[i][j] = 0

    return binary_matrix

# GPU training
GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True                         # start ncuda
    torch.backends.cudnn.benchmark = True                       # cuda gpu automation calculation
    dtype = torch.cuda.FloatTensor                              # cuda operation type of float
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'                    # env variable
    print("num GPUs",torch.cuda.device_count())                 # gpu avaiable
else:
    dtype = torch.FloatTensor                            

# device=torch.device('cuda:0' if GPU else 'cpu')
device=torch.device('cuda:0' if GPU else 'cpu')

#定义参数
img_W = 128
img_H = 80
SR = 1500 / (128 * 80)                                          # sampling rate
batch_size = 1                                
lr0 = 0.05                                                      # learning rate
TV_strength = 1e-9                                              # regularization parameter of Total Variation
num_patterns = int(np.round(img_W * img_H * SR))                # number of measurement times  
Steps = 251                                                     # optimization steps                              
decay_rate = 0.90
decay_steps = 100
alpha = 0.2
alpha_step = 1 / 400
# alpha_step = 1/(Steps-1)

target = 'hellokitty'            # 7, C, L, N, O, P
# cars123,doraemon,ghost123,hellokitty,npu,panda12.rabbit,sunflower.shuttlecock,whale,words
position1 = '0000'
position2 = '0450'
position3 = '0900'
position4 = '1225'                                                                                                                                                                 
position5 = '0000'
position6 = '0225'
position7 = '0450'
speckle_size = '060'
wave_plate2 = 'wp2'
wave_plate4 = 'wp4'
number_patterns = 1500
rotation = position1
wave_plate = wave_plate2

result_save_path = '/data/sz3/temp/result/MPFNet-simulating results-128x80/'
# patterns = np.load('/data/sz3/temp/patterns/npy_resized_64x64/sp_%s.npy'%(speckle_size))
patterns = np.load('/data/sz3/temp/patterns/npy_resized_128x80/%s_128x80.npy'%(speckle_size))

# y = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/ground truth/020/2_020.txt')                                         # ground truth list

# y2_1 = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/2024-3-6-020/2_wp2_r0000_s020_gt.txt')        # wp2-r0000 polarized list
# y2_2 = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/2024-3-6-020/0_wp2_r0450_s100_gt.txt')        # wp2-r0450 polarized list

# data_template = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/2/2_wp2_r0000_s040_uw.txt')
# data1_uw = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/%s/%s_wp2_r0000_s%s_uw.txt'%(target, target, speckle_size))
# data2_uw = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/%s/%s_wp2_r0450_s%s_uw.txt'%(target, target, speckle_size))
# # data3_uw = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/%s/%s_wp2_r0900_s%s_uw.txt'%(target, target, speckle_size))
# # data4_uw = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/%s/%s_wp2_r1225_s%s_uw.txt'%(target, target, speckle_size))
# # data5_uw = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/%s/%s_wp2_r0000_s%s_uw.txt'%(target, target, speckle_size))
# # data6_uw = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/%s/%s_wp2_r0225_s%s_uw.txt'%(target, target, speckle_size))
# # data7_uw = np.loadtxt('/data/lx-2022/MICPNet/Network-UNet/Dense-PSP-UNet/data/underwater experiment data I/%s/%s_wp2_r0450_s%s_uw.txt'%(target, target, speckle_size))
#
# y = (data1_uw - data2_uw/3)
# y1 = data1_uw
# y2 = data2_uw
# data1_uw = np.loadtxt('/data/sz3/temp/data/polarizaton data 2024-5-16/%s_wp2_r0450_s%s_uw.txt'%(target, speckle_size))
data1_uw = np.loadtxt('/data/sz3/temp/data/bkd_lx 2024-10-12/%s.txt'%(target))

y = data1_uw

# DGI reconstruction
print('DGI reconstruction...')
B_aver  = 0
SI_aver = 0
R_aver = 0 
RI_aver = 0
count = 0
for i in range(num_patterns):    
    pattern = patterns[i, :, :] 
    B_r = y[i]  
    count = count + 1
    SI_aver = (SI_aver * (count - 1) + pattern * B_r)/count 
    B_aver  = (B_aver * (count - 1) + B_r) / count 
    R_aver = (R_aver * (count - 1) + sum(sum(pattern))) / count 
    RI_aver = (RI_aver * (count -1) + sum(sum(pattern)) * pattern) / count 
    DGI = SI_aver - B_aver / R_aver * RI_aver
    
    # print(i)
        
    # if i % 250 == 0 and i != 0:  
    #     DGI_temp1 = DGI.T
    #     DGI_temp1 = 255 - DGI
    #     DGI_temp1 = DGI_temp1 - np.min(DGI_temp1)
    #     DGI_temp1 = DGI_temp1 * 255 / np.max(np.max(DGI_temp1))
            
    #     DGI_temp1_numpy = DGI_temp1
            
    #     DGI_temp1 = Image.fromarray(DGI_temp1.astype('uint8')).convert('L')
    #     DGI_temp1.save(result_save_path + '%s_TGI_n%d_s%s.bmp' % (target, i, speckle_size))
# DGI[DGI<0] = 0
print('Finished')

# region Data Transformation for Speckle Patterns
A_real = torch.tensor(patterns[0 : num_patterns, :, :]).float().to(device)
mean_a, variance_a = torch.mean(A_real), torch.var(A_real)
A_real = (A_real - mean_a) / torch.sqrt(variance_a)
# endregion


model1 = GIDC().cuda()
# model1 = ResUNet().cuda()

_initialize_weights(model1)
l1_regularization = 0
l2_regularization = 0
for param in model1.parameters():
    l1_regularization += torch.abs(param).sum()
    l2_regularization += (param ** 2).sum()

fc_layer = nn.Linear(num_patterns, img_H * img_W).to(device)

#input 1
y_real = torch.tensor(y[0 : num_patterns])                                  
y_real = torch.reshape(y_real, (1, num_patterns, 1, 1)).float().to(device)
mean_yr, variance_ya = torch.mean(y_real), torch.var(y_real)
y_real = (y_real - mean_yr) / torch.sqrt(variance_ya)

input1 = torch.reshape(y_real, (1, 1, 1, num_patterns)).float().to(device)
input1 = fc_layer(input1)
input1 = torch.reshape(input1, (1, 1, img_W, img_H)).float().to(device)

# # input 2 - 方法一，输入2是垂直分量
# y_added = torch.tensor(y2[0 : num_patterns])
# y_added = torch.reshape(y_added, (1, num_patterns, 1, 1)).float().to(device)
# mean_ya, variance_ya = torch.mean(y_added), torch.var(y_added)
# y_added = (y_added - mean_ya) / torch.sqrt(variance_ya)

# input2 = torch.reshape(y_added, (1, 1, 1, num_patterns)).float().to(device)
# input2 = fc_layer(input2)
# input2 = torch.reshape(input2, (1, 1, img_W, img_H)).float().to(device)

# input 2
input2 = torch.randn(1, 1, img_W, img_H).to(device).float()

DGI = (DGI - np.mean(DGI)) / np.std(DGI) 
DGI = np.reshape(DGI, [img_W, img_H], order='F') 
DGI = torch.tensor(DGI)

# input2 = torch.reshape(DGI, (1, 1, img_W, img_H)).to(device).float()
# input2 = torch.randn(1, 1, img_W, img_H).to(device).float()

parameters = list(model1.parameters()) + list(fc_layer.parameters())
optimizer1 = optim.Adam(parameters, lr = lr0, betas=(0.5, 0.999), eps = 1e-08)
scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=decay_rate, last_epoch=-1, verbose=False)

# region Different Optimizer
# optimizer2 = optim.Adam(model2.parameters(),lr=lr0,betas=(0.5, 0.9),eps=1e-08)
# scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=decay_rate, last_epoch=-1, verbose=False)

# for epoch in range(Steps):
    
# 方案1不同权值
    # 前向传播
    # model.train()
    # x_pred1,y_pred1 = model(input1,A_real,img_W,img_H,num_patterns)
    # x_pred2,y_pred2 = model(input2,A_real,img_W,img_H,num_patterns)
    # TV_reg = TV_strength *tv_regularization(x_pred1)
    # loss_y1 = loss_fn(y_real, y_pred1) 
    # loss1 = loss_y1+TV_reg 
    # loss1 = loss_y1
    # TV_reg = TV_strength *tv_regularization(x_pred2)
    # loss_y2 = loss_fn(y_real, y_pred2) 
    # loss2 = loss_y2+TV_reg 
    # loss2 = loss_y2
    # loss_diff = torch.mean(torch.abs(x_pred1-x_pred2))
    # # loss = alpha * loss_diff + ((1-alpha)/2)* (loss1 + loss2)
    # loss = alpha * loss_diff + (1-alpha)*(loss1 + loss2)
    # alpha = alpha + alpha_step

    # #反向传播和优化
    # optimizer.zero_grad()
    # # optimizer2.zero_grad()
    # loss.backward(retain_graph=True)
    # optimizer.step()
    # # optimizer2.step()
    # endregion

for epoch in range(Steps):
    model1.train()
    
    x_pred1, y_pred1 = model1(input1.detach(), A_real, img_W, img_H, num_patterns)
    x_pred2, y_pred2 = model1(input2.detach(), A_real, img_W, img_H, num_patterns)
    
    # y_real loss
    TV_reg = TV_strength * tv_regularization(x_pred1)
    # loss_y1 = loss_fn(y_real, -y_pred1) 
    loss_y1 = loss_fn(y_real, y_pred1)
    loss1 = loss_y1 + TV_reg 
    # y_added loss
    TV_reg = TV_strength * tv_regularization(x_pred2)
    # loss_y2 = loss_fn(y_added, -y_pred2) 
    loss_y2 = loss_fn(y_real, y_pred2)
    loss2 = loss_y2 + TV_reg
    
    loss_diff = torch.mean(torch.abs(x_pred1 - x_pred2))
    loss = alpha * loss_diff + (1 - alpha) * (loss1 + loss2)
    if alpha >= 0.8:
        loss = 0.8 * loss_diff + (1 - alpha) * (loss1 + loss2)
    else:
        alpha = alpha + alpha_step

    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

    if epoch%10 == 0: 
            print('step:%d----y loss:%f:' % (epoch,loss)) 
            DGI_temp0 = np.reshape(DGI, (img_H,img_W)) 
            # region Other Training Code
            # plt.imshow(DGI_temp0)
            # plt.title('DGI')
            # plt.yticks([])

            # plt.subplot(142)
            # x = torch.reshape(x_pred1,(img_H,img_W))
            # pltx = x.cpu().detach().numpy()
            # plt.imshow(pltx)
            # plt.title('pred1')
            # plt.yticks([])
            
            # plt.subplot(143)
            # py = torch.reshape(x_pred2,(img_W,img_H))
            # pltpy = py.cpu().detach().numpy()
            # plt.imshow(pltpy)
            # plt.title('pred2')     
            # plt.yticks([])

            # ax1 = plt.subplot(143)
            # py = torch.reshape(y_pred1,(num_patterns,1))
            # pltpy = py.cpu().detach().numpy()
            # plt.plot(pltpy)
            # plt.title('pred_y')
            # ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')       
            # plt.yticks([])
            

            # plt.subplots_adjust(hspace=0.25, wspace=0.25)
            # plt.show()
            
            # x_pred = (x_pred1+x_pred2)/2
            # x_pred = x_pred - torch.min(x_pred)
            # x_pred = x_pred*255/torch.max(torch.max(x_pred))
            # x_pred = torch.reshape(x_pred,(img_H,img_W))
            # # x_pred1 = x_pred1.T
            # # x_pred1 = binarize_matrix(x_pred1,50)
            # # x_pred1= torch.tensor(x_pred1)
            # x_pred = Image.fromarray(x_pred.detach().cpu().numpy().astype('uint8')).convert('L')
            # x_pred.save(result_save_path + 'P_%d_%d.bmp'%(num_patterns,epoch))

            # a = 1.1
            # x_pred1 = x_pred1 - torch.min(x_pred1)
            # x_pred1 = x_pred1*255/torch.max(torch.max(x_pred1))
            # x_pred1 = torch.reshape(x_pred1,(img_H,img_W))
            # x_pred1 = x_pred1.detach().cpu().numpy()
            # # x_pred1 = x_pred1.T
            # # x_pred1 = binarize_matrix(x_pred1,50)
            # # x_pred1= torch.tensor(x_pred1)
            # # x_pred1 = np.clip(a * x_pred1, 0, 250).astype(np.uint8)
            # x_pred1 = Image.fromarray(x_pred1.astype('uint8')).convert('L')
            # x_pred1.save(result_save_path + 'N_%d_%d_1.bmp'%(num_patterns,epoch))
            # endregion

            # x_pred2 = 255 - x_pred2
            x_pred2 = x_pred2 - torch.min(x_pred2)
            x_pred2 = x_pred2 * 255 / torch.max(torch.max(x_pred2))
            x_pred2 = torch.reshape(x_pred2, (img_H, img_W))
            x_pred2 = x_pred2.T
            # x_pred1 = binarize_matrix(x_pred1,50)
            # x_pred1= torch.tensor(x_pred1)
            x_pred2 = Image.fromarray(x_pred2.detach().cpu().numpy().astype('uint8')).convert('L')
            x_pred2.save(result_save_path + 'MPFNet_%s_s%s_n%d_e%d_uw_CP.bmp' % (target, speckle_size, num_patterns, epoch))

            # DGI_temp0 = 255 - DGI_temp0
            DGI_temp0 = DGI_temp0 - torch.min(DGI_temp0)
            DGI_temp0 = DGI_temp0*255 / torch.max(torch.max(DGI_temp0))
            DGI_temp0 = torch.reshape(DGI_temp0, (img_H, img_W))
            x_pred1 = x_pred1.T
            DGI_temp0 = Image.fromarray(DGI_temp0.detach().cpu().numpy().astype('uint8')).convert('L')
            DGI_temp0.save(result_save_path + 'MPFNet_%s_n%d_PCGI_uw_CP.bmp'%(target, num_patterns))
            
            # x_pred1 = x_pred1 - torch.min(x_pred1)
            # x_pred1 = x_pred1*255/torch.max(torch.max(x_pred1))
            # x_pred1 = torch.reshape(x_pred1,(img_W,img_H))
            # # x_pred2 = x_pred2.Tc
            # x_pred1 = Image.fromarray(x_pred1.detach().cpu().numpy().astype('uint8')).onvert('L')
            # x_pred1.save(result_save_path + 'GIDC_%d_%d.bmp'%(num_patterns,epoch))

# torch.save(model1.state_dict(), 'model1.pth')