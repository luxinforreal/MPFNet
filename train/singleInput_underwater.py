import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_ssim
import torch.nn.functional as F

import sys
sys.path.append('/data/sz3/temp')

from scipy.io import loadmat
# from testingpart import _initialize_weights, MICPNet, AttentionModule, PolarizationNetwork
# from testingMICPNet import _initialize_weights, MICPNet, AttentionModule, PolarizationNetwork
# from BasicPolarizationAttentionNetwork import BasicPolarizationAttentionNetwork, _initialize_weights
# from BasicPolarizationNetwork import BasicPolarizationNetwork, _initialize_weights
# from BasicGIDCNetwork import BasicGIDCNetwork, _initialize_weights
# from BasicMICPNet import BasicMICPNet, _initialize_weights
# from BasicResUNet import BasicResUNet, _initialize_weights
# from BasicSimpleNet import BasicSimpleNet, _initialize_weights
from model.GIDC import GIDC, _initialize_weights
from model.ResUNet import ResUNet, _initialize_weights
from model.MPFNet import MPFNet, _initialize_weights
from torchsummary import summary
from PIL import Image
# from ResNet.ResNet_ImageNet import ACmix_ResNet
# from res_net import UNet
# from GIDC_0 import UNet0

torch.cuda.empty_cache()

# wtf

seed = 99
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#定义TV损失和LOSS
def tv_regularization(img):
    dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return torch.mean(dx) + torch.mean(dy)

def loss_fn(y, y_pred):
    return torch.mean(torch.square(y - y_pred))

def loss_ssim(image1,image2):
    return pytorch_ssim.ssim(image1,image2) 

def psnr(target, prediction):
    mse = F.mse_loss(target, prediction)                    # MSE
    max_pixel = torch.max(target)                           # MAX_PIXEL
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))    # PSNR
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

# 选择GPU进行训练
GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True                 # start cuda
    torch.backends.cudnn.benchmark = True               # cuda automaticallt optimizes GPU calculations
    dtype = torch.cuda.FloatTensor                      # opareation on cuda is float type
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"      # visiable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print("num GPUs",torch.cuda.device_count())         # avaliable GPU
else:
    dtype = torch.FloatTensor                           # float operation on CPU if GPU unavaliable

# device=torch.device('cuda:0' if GPU else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# region Define Parameters
img_W = 128
img_H = 80
SR = 1500 / (128 * 80)                                   # sampling rate
batch_size = 1                                
lr0 = 0.04                                              # learning rate
TV_strength = 1e-9                                      # regularization parameter of Total Variation
# num_patterns = int(np.round(img_W * img_H * SR))      # number of measurement times  
num_patterns = 1500
Steps = 231                                             # optimization steps                              
decay_rate = 0.90
decay_steps = 100
# endregion


target = 'ghost3'                            # 7,N,C,L,O,P
                                        # cars123,doraemon,ghost123,hellokitty,npu,panda12.rabbit,sunflower.shuttlecock,whale,words
position1 = '0000'
position2 = '0450'
position3 = '0900'
position4 = '1225'                                                                                                                                                                 
position5 = '0000'
position6 = '0225'
position7 = '0450'
speckle_size = '060'
number_patterns = 1500
wave_number = '4'
# result_save_path = '/data/sz3/temp/result/GIDC-128x80-r0000/'
# result_save_path = '/data/sz3/temp/result/ResUNet/'
# result_save_path = '/data/sz3/temp/result/GIDC-simulating results-128x80/'
result_save_path = '/data/sz3/temp/result/GIDC-simulating results-128x80/'


# patterns = np.load('/data/sz3/temp/patterns/npy_resized_64x64/sp_%s.npy'%(speckle_size))
patterns = np.load('/data/sz3/temp/patterns/npy_resized_128x80/%s_128x80.npy'%(speckle_size))
# data1_uw = np.loadtxt('/data/sz3/temp/data/polarizaton data 2024-5-16/%s_wp2_r0000_s%s_uw.txt'%(target, speckle_size))
data1_uw = np.loadtxt('/data/sz3/temp/data/bkd_lx 2024-10-12/%s.txt'%(target))
rotation = position1
y = data1_uw

# region DGI reconstruction
print('DGI reconstruction...')
B_aver  = 0
SI_aver = 0
R_aver = 0 
RI_aver = 0
count = 0
for i in range(num_patterns):    
    pattern = patterns[i,:,:] 
    B_r = y[i]  
    count = count + 1
    SI_aver = (SI_aver * (count -1) + pattern * B_r)/count 
    B_aver  = (B_aver * (count -1) + B_r)/count 
    R_aver = (R_aver * (count -1) + sum(sum(pattern)))/count 
    RI_aver = (RI_aver * (count -1) + sum(sum(pattern))*pattern)/count 
    DGI = SI_aver - B_aver / R_aver * RI_aver
# DGI[DGI<0] = 0
del B_aver, SI_aver, R_aver, RI_aver, count 
print('Finished')
# endregion DGI

# region Data Transformation for single Input
y_test = torch.tensor(y[0:num_patterns])
y_test = torch.reshape(y_test, (1,num_patterns, 1, 1)).float().cuda()
mean_ya, variance_ya = torch.mean(y_test), torch.var(y_test)
y_test = (y_test - mean_ya) / torch.sqrt(variance_ya)

A_real = torch.tensor(patterns[0:num_patterns, :, :]).float().cuda()
mean_a, variance_a = torch.mean(A_real), torch.var(A_real)
A_real = (A_real - mean_a) / torch.sqrt(variance_a)
print("A_real:", A_real.size())

# Transfor the construction
fc_layer = nn.Linear(num_patterns, img_H * img_W).cuda()

input_test = torch.reshape(y_test, (1, 1, 1, num_patterns)).float().cuda()
input_test = fc_layer(input_test)
input_test = torch.reshape(input_test, (1, 1, img_W, img_H)).float().cuda()
input_test = torch.randn((1, 1, img_W, img_H)).float().cuda()

model = MPFNet().cuda()
# model = ResUNet().cuda()

_initialize_weights(model)
l1_regularization = 0
l2_regularization = 0
for param in model.parameters():
    l1_regularization += torch.abs(param).sum()
    l2_regularization += (param ** 2).sum()
    
DGI = (DGI - np.mean(DGI))/np.std(DGI) 
DGI = np.reshape(DGI, [img_W, img_H], order='F') 
DGI = torch.tensor(DGI)

parameters = list(model.parameters())
optimizer = optim.Adam(parameters,lr = lr0, betas=(0.5, 0.9), eps=1e-08)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate, last_epoch=-1, verbose=False)



for epoch in range(Steps):
    
# 方案1不同权值
# s前向传播
    model.train()
    # x_pred,y_pred = model(input1,input2,A_real,img_W,img_H,num_patterns)
    x_pred, y_pred = model(input_test.detach(), A_real, img_W, img_H, num_patterns)
    TV_reg = TV_strength * tv_regularization(x_pred)
    # loss_y = loss_fn(y_test, (-y_pred))        # ground truth training
    loss_y = loss_fn(y_test, (y_pred))          # underwater data training
    loss = loss_y + TV_reg 
    # loss = loss_y

    #反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0: 
            print('step:%d-----------y loss:%f:' % (epoch,loss)) 

            x_pred = x_pred - torch.min(x_pred)
            x_pred = x_pred * 255 / torch.max(torch.max(x_pred))
            x_pred = torch.reshape(x_pred, (img_H, img_W))
            # x_pred = 255 - x_pred   # 数据反向
            x_pred = x_pred.T

            x_pred = Image.fromarray(x_pred.detach().cpu().numpy().astype('uint8')).convert('L')
            # x_pred.save(result_save_path + '%s_s%s_n%d_e%d_gt.bmp' % (target, speckle_size, num_patterns, epoch))
            x_pred = x_pred.transpose(Image.FLIP_LEFT_RIGHT)
            x_pred.save(result_save_path + 'GIDC_%s_wp%s_r%s_s%s_n%d_e%d_uw.bmp' % (target, wave_number, rotation, speckle_size, num_patterns, epoch))

            DGI_temp0 = DGI
            # DGI_temp0 = 255 - DGI   # 数据反向
            DGI_temp0 = np.reshape(DGI_temp0, (img_H, img_W)) 
            DGI_temp0 = DGI_temp0 - torch.min(DGI_temp0)
            DGI_temp0 = DGI_temp0 * 255 / torch.max(torch.max(DGI_temp0))
            DGI_temp0 = torch.reshape(DGI_temp0, (img_H, img_W))
            # x_pred1 = x_pred1.T
            DGI_temp0 = Image.fromarray(DGI_temp0.detach().cpu().numpy().astype('uint8')).convert('L')
            # DGI_temp0.save(result_save_path + '%s_n%d_DGI.bmp'%(target, num_patterns))
            x_pred = x_pred.transpose(Image.FLIP_LEFT_RIGHT)
            DGI_temp0.save(result_save_path + 'GIDC_%s_wp%s_r%s_n%d_DGI.bmp'%(target, wave_number, rotation, num_patterns))
            # represents the DGI result  
