import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
Relu = 0.01


def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)  

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(Relu),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class MPFNet(nn.Module):
    def __init__(self):
        super(MPFNet,self).__init__()
        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(Relu),
        )
        self.layer1 = self.make_layer(ResBlock,inchannel=16, outchannel=32, num_blocks=2, stride=1)
        self.conv_pooling_1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(Relu),
        )
        self.layer2 = self.make_layer(ResBlock,inchannel=32, outchannel=64, num_blocks=2, stride=1)
        self.conv_pooling_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(Relu),
        )
        self.layer3 = self.make_layer(ResBlock,inchannel=64, outchannel=128, num_blocks=2, stride=1)
        self.conv_pooling_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(Relu),
        )
        self.layer4 = self.make_layer(ResBlock,inchannel=128, outchannel=256, num_blocks=2, stride=1)
        self.conv_pooling_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(Relu),
        )
        # self.layer5 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer5 = self.make_layer(ResBlock,inchannel=256, outchannel=512, num_blocks=2, stride=1)
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(Relu),
        )
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(Relu),
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(Relu),
        )
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(Relu),
        )
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(Relu),
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(Relu),
        )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(Relu),
        )
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(Relu),
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(Relu),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(Relu),
        )
        self.conv9_1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(Relu),
        )
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(Relu),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=1, kernel_size=5,stride=1,padding='same'),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(Relu),
        )
        
        # region added the channel attention block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(256, 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 256, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial_conv1 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)
        # endregion

    # def make_layer(self, block, channels, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.inchannel, channels, stride))
    #         self.inchannel = channels
    #     return nn.Sequential(*layers)
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(inchannel, outchannel, stride))
            return nn.Sequential(*layers)
    
    
    def forward(self, x, A_real, img_W, img_H, num_patterns):
    # def forward(self,x):
        # conv0 = self.conv0(x)
        # conv1 = self.layer1(conv0)
        # conv2 = self.layer2(conv1)
        # conv3 = self.layer3(conv2)
        # conv4 = self.layer4(conv3)
        # conv5 = self.layer5(conv4)
        # conv6 = self.conv6(conv5)s
        
        conv0 = self.conv0(x)
        conv1_1 = self.layer1(conv0)
        pool_1 = self.conv_pooling_1(conv1_1)
        conv2_1 = self.layer2(pool_1)
        pool_2 = self.conv_pooling_2(conv2_1)
        conv3_1 = self.layer3(pool_2)
        pool_3 = self.conv_pooling_3(conv3_1)
        conv4_1 = self.layer4(pool_3)
        pool_4 = self.conv_pooling_4(conv4_1)
        conv5_1 = self.layer5(pool_4)
        conv6 = self.conv6(conv5_1)

        # region Add MBF Blocks

        # region Add Attention Blocks
            

        merge1 = torch.cat([conv4_1,conv6], axis = 1)
        conv6_1 = self.conv6_1(merge1)
        conv6_2 = self.conv6_2(conv6_1)
        conv7 = self.conv7(conv6_2)
        merge2 = torch.cat([conv3_1,conv7], axis = 1)
        conv7_1 = self.conv7_1(merge2)
        conv7_2 = self.conv7_2(conv7_1)
        conv8 = self.conv8(conv7_2)
        merge3 = torch.cat([conv2_1,conv8], axis = 1)
        conv8_1 = self.conv8_1(merge3)
        conv8_2 = self.conv8_2(conv8_1)
        conv9 = self.conv9(conv8_2)
        merge4 = torch.cat([conv1_1,conv9], axis = 1)
        conv9_1 = self.conv9_1(merge4)
        conv9_2 = self.conv9_2(conv9_1)
        
        x_out = self.conv10(conv9_2)
        x_out = x_out / torch.max(x_out)
        
        y_out = torch.empty(1,0,1,1).cuda()
        
        for pattern in A_real:
            # pattern = torch.reshape(pattern,(img_W,img_H))
            # pattern = pattern.T
            pattern = torch.reshape(pattern,(1,1,img_W,img_H))
            conv_result = F.conv2d(x_out, pattern, stride=1, padding=0)
            y_out=torch.cat((y_out,conv_result),dim=1)
            
        x_out = torch.reshape(x_out,(1,1,img_W,img_H))
        y_out = torch.reshape(y_out,(1,num_patterns,1,1))
        
        mean_x, variance_x = torch.mean(x_out), torch.var(x_out)
        mean_y, variance_y = torch.mean(y_out), torch.var(y_out)
        
        x_pred = (x_out - mean_x) / torch.sqrt(variance_x)
        y_pred = (y_out - mean_y) / torch.sqrt(variance_y)
        
        return x_pred, y_pred
    
    
    