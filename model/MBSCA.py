import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        attn = self.bn(attn)
        return x * torch.sigmoid(attn)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
    
    def forward(self, x):
        avg_out = F.adaptive_avg_pool2d(x, 1)
        max_out = F.adaptive_max_pool2d(x, 1)
        attn = self.fc1(avg_out) + self.fc1(max_out)
        attn = F.relu(attn)
        attn = self.fc2(attn)
        return x * torch.sigmoid(attn)

class MBSCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.spatial_att = SpatialAttention()
        self.channel_att = ChannelAttention(in_channels * 3)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.final_conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, feat_a, feat_b, feat_c):
        feat_ab = self.relu(self.bn(self.conv1x1(torch.cat([feat_a, feat_b], dim=1))))
        att_ab = self.spatial_att(feat_ab)
        
        feat_bc = self.relu(self.bn(self.conv1x1(torch.cat([feat_b, feat_c], dim=1))))
        att_bc = self.spatial_att(feat_bc)
        
        feat_abc = self.relu(self.bn(self.conv1x1(torch.cat([feat_a, feat_b, feat_c], dim=1))))
        att_abc = self.channel_att(feat_abc)
        
        if att_ab.shape != att_bc.shape:
            raise ValueError("Shape mismatch between att_ab and att_bc")
        
        fusion_bc = self.fusion_conv(torch.cat([att_ab, att_bc], dim=1)).mean(dim=[2,3], keepdim=True)
        
        if att_abc.shape[-2:] != fusion_bc.shape[-2:]:
            raise ValueError("Shape mismatch between att_abc and fusion_bc")
        
        final_feat = torch.cat([att_abc, fusion_bc.expand_as(att_abc)], dim=1)
        final_feat = self.relu(self.final_bn(self.final_conv(final_feat)))
        return final_feat

# testing
# def test_module():
#     batch_size, C, H, W = 1, 32, 32, 32
#     feat_a = torch.randn(batch_size, C, H, W)
#     feat_b = torch.randn(batch_size, C, H, W)
#     feat_c = torch.randn(batch_size, C, H, W)
    
#     model = FusionModule(in_channels=C, out_channels=64)
#     output = model(feat_a, feat_b, feat_c)
#     print("Output shape:", output.shape)

# test_module()
