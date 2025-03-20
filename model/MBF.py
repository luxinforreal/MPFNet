import torch
import torch.nn as nn

class MultiBranchFeatureAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_size):
        """
        :param in_channels_list: 输入通道数的列表 [C1, C2, C3]，分别对应 input1, input2, input3
        :param out_channels: 统一输出的通道数
        :param target_size: (H_1, W_1) 目标特征图大小
        """
        super(MultiBranchFeatureAlign, self).__init__()
        self.target_size = target_size
        
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_c in in_channels_list
        ])

    def forward(self, input1, input2, input3):
        """
        :param input1: Tensor 形状 (B1, C1, H1, W1)
        :param input2: Tensor 形状 (B2, C2, H2, W2)
        :param input3: Tensor 形状 (B3, C3, H3, W3)
        :return: 拼接后的特征图，形状 (B4, 3*out_channels, H_1, W_1)
        """
        inputs = [input1, input2, input3]
        aligned_features = []

        for i, x in enumerate(inputs):
            x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
            x = self.transforms[i](x)
            aligned_features.append(x)

        fused = torch.cat(aligned_features, dim=1)
        return fused
