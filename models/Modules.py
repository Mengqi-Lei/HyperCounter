import torch
from torch import nn


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class DenseFusion(nn.Module):
    def __init__(self, in_C, out_C):
        super(DenseFusion, self).__init__()
        down_factor = in_C // out_C
        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main1 = BasicConv2d(in_C,out_C,kernel_size=1)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        feat = self.fuse_down_mul(rgb + depth) 
        return self.fuse_main(self.res_main(feat) + feat)


class Resudiual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resudiual, self).__init__()
        self.conv = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.sigmoid(x1)
        out = x1 * x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x

class BasicUpsample(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            nn.Conv2d(32,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)

class FeatureFusionAndPrediction(nn.Module):
    def __init__(self,):
        super(FeatureFusionAndPrediction, self).__init__()
        self.basicconv1 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=1)
        self.basicconv2 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=1)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(32,32,1),
            nn.ReLU()
        )
        self.basicconv3 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv4 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=3,stride=1,padding=1)
        
        self.basicupsample32 = BasicUpsample(scale_factor=32)
        self.basicupsample16 = BasicUpsample(scale_factor=16)
        self.basicupsample8 = BasicUpsample(scale_factor=8)
        self.basicupsample4 = BasicUpsample(scale_factor=4)


        self.reg_layer = nn.Sequential(
            nn.Conv2d(128, 64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
            )

    def forward(self,out_data_4,out_data_8,out_data_16,out_data_32):

        
        out_data_32 = self.basicconv1(out_data_32)
        out_data_32 = self.basicconv3(out_data_32)
        
        out_data_16 = self.basicconv1(out_data_16)  
        out_data_16 = torch.cat([out_data_16,self.upsample1(out_data_32)],dim=1)
        out_data_16 = self.basicconv4(out_data_16)

        out_data_8 = self.basicconv1(out_data_8)
        out_data_8 = torch.cat([out_data_8,self.upsample1(out_data_16)],dim=1)
        out_data_8 = self.basicconv4(out_data_8)

        out_data_4 = self.basicconv1(out_data_4)
        out_data_4 = torch.cat([out_data_4,self.upsample1(out_data_8)],dim=1)
        out_data_4 = self.basicconv4(out_data_4)

        out_data_32 = self.basicupsample32(out_data_32)
        out_data_16 = self.basicupsample16(out_data_16) 
        out_data_8 = self.basicupsample8(out_data_8)    
        out_data_4 = self.basicupsample4(out_data_4)    

        out_data = torch.cat([out_data_32,out_data_16,out_data_8,out_data_4],dim=1) 

        out_data = self.reg_layer(out_data)


        return torch.abs(out_data)




