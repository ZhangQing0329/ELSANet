import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

# from .deeplab_resnet import resnet50_locate
# from .vgg import vgg16_locate
from torchvision.models.resnet import resnet50
# RAS_33
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out6 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out6, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out6, out2, out3, out4, out5

    def initialize(self):
        res50 = models.resnet50(pretrained=True)
        self.load_state_dict(res50.state_dict(), False)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        r50 = resnet50(True)
        self.cv1 = r50.conv1
        self.bn1 = r50.bn1
        self.mxp = r50.maxpool
        self.re1 = r50.relu
        self.layer1 = r50.layer1
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        self.layer4 = r50.layer4

    def forward(self, x):
        x1 = self.mxp(self.re1(self.bn1(self.cv1(x))))
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


class ASPP_246(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP_246, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


# edge-guided interaction module
class EGI(nn.Module):
    def __init__(self,in_channel=64, ratio=2):
        super(EGI, self).__init__()

        self.conv_query = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, sod, edge):
        if edge.size()[2:] != sod.size()[2:]:
            edge = F.interpolate(edge, size=sod.size()[2:], mode='bilinear', align_corners=True)
        bz,c,h,w=sod.shape

        edge_q = self.conv_query(edge).view(bz, -1, h * w).permute(0, 2, 1)
        edge_k = self.conv_key(edge).view(bz, -1, h * w)
        mask = torch.bmm(edge_q, edge_k)  # bz, hw, hw
        mask = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(sod).view(bz, c, -1)
        feat = torch.bmm(rgb_v, mask.permute(0, 2, 1))  # bz, c, hw
        feat = feat.view(bz, c, h, w)
        out = sod + feat * sod

        return out


# Low-level Weighted Fusion Module
class LWF(nn.Module):
    def __init__(self, channels=64, r=2):
        super(LWF, self).__init__()

        inter_channels = int(channels // r)

        self.sigmoid = nn.Sigmoid()

        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v = nn.BatchNorm2d(64)

        self.conv_x = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.bn_x = nn.BatchNorm2d(64)

        self.convfuse = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnfuse = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=True)
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        mul = torch.mul(out1h, out1v)
        fuse = torch.cat((out1h, out1v), 1)
        fuse = torch.cat((fuse, mul), 1)
        x = F.relu(self.bn_x(self.conv_x(fuse)), inplace=True)

        wei = self.sigmoid(x)
        fuse_channel = out1h * wei + out1v * (1 - wei)

        out3h = F.relu(self.bn3h(self.conv3h(fuse_channel)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse_channel)), inplace=True)

        out4h = F.relu(self.bn4h(self.conv4h(out3h + out1h-out3h*out1h)), inplace=True)
        out4v = F.relu(self.bn4v(self.conv4v(out3v + out1v-out3v*out1v)), inplace=True)

        out = torch.cat((out4h, out4v), 1)
        out = F.relu(self.bnfuse(self.convfuse(out)), inplace=True)
        return out


class DualDecoder(nn.Module):
    def __init__(self):
        super(DualDecoder, self).__init__()
        # sod
        self.s34 = LWF()
        self.s23 = LWF()
        self.s12 = LWF()
        # egi
        self.egi3 = EGI(64,2)
        self.egi2 = EGI(64,2)
        self.egi1 = EGI(64,2)
        # edge
        self.e34 = LWF()
        self.e23 = LWF()
        self.e12 = LWF()

    def forward(self, out1s, out2s, out3s,out1d, out2d, out3d, out_h):
        # edge branch
        out3d = self.e34(out3d, out_h)
        out2d = self.e23(out2d, out3d)
        out1d = self.e12(out1d, out2d)

        # saliency branch: stage3
        out3s = self.s34(out3s, out_h)
        cmf3s = self.egi3(out3s, out3d)
        # saliency branch: stage2
        out2s = self.s23(out2s, cmf3s)
        cmf2s = self.egi2(out2s, out2d)
        # saliency branch: stage1
        out1s = self.s12(out1s, cmf2s)
        cmf1s = self.egi1(out1s, out1d)

        return cmf1s,out1s, out2s, out3s, out1d,out2d,out3d


# High-level Interactive Fusion Module
class HIF(nn.Module):
    def __init__(self, in_channels=64, squeeze_ratio=2):
        super(HIF, self).__init__()
        inter_channels = in_channels // squeeze_ratio  # reduce computation load
        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv_q1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.delta1 = nn.Parameter(torch.Tensor([0.1]))  # initiate as 0.1
        self.delta2 = nn.Parameter(torch.Tensor([0.1]))  # initiate as 0.1
        self.delta3 = nn.Parameter(torch.Tensor([0.1]))  # initiate as 0.1
        self.delta4 = nn.Parameter(torch.Tensor([0.1]))  # initiate as 0.1

        self.convfuse = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnfuse = nn.BatchNorm2d(64)

    def forward(self, left,down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=True)
        B, C, H, W = left.size()
        P = H * W
        left_q = self.conv_q(left).view(B, -1, P).permute(0, 2, 1)  # [B, P, C']
        left_k = self.conv_k(left).view(B, -1, P)  # [B, C', P]
        left_v = self.conv_v(left).view(B, -1, P)  # [B, C, P]
        left_weights = F.softmax(torch.bmm(left_q, left_k), dim=1)  # column-wise softmax, [B, P, P]
        left_self = torch.bmm(left_v, left_weights).view(B, C, H, W)

        down_q = self.conv_q1(down).view(B, -1, P).permute(0, 2, 1)  # [B, P, C']
        down_k = self.conv_k1(down).view(B, -1, P)  # [B, C', P]
        down_v = self.conv_v1(down).view(B, -1, P)  # [B, C, P]
        down_weights = F.softmax(torch.bmm(down_q, down_k), dim=1)  # column-wise softmax, [B, P, P]
        down_self = torch.bmm(down_v, down_weights).view(B, C, H, W)

        left_other = torch.bmm(left_v, down_weights).view(B, C, H, W)
        down_other = torch.bmm(down_v, left_weights).view(B, C, H, W)

        out1 = self.delta1 * left_self + self.delta2 * left_other +left
        out2 = self.delta3 * down_self + self.delta4 * down_other + down
        out = torch.cat((out1, out2), 1)
        out = F.relu(self.bnfuse(self.convfuse(out)), inplace=True)
        return out

class ELSANet(nn.Module):
    def __init__(self, cfg):
        super(ELSANet, self).__init__()
        self.cfg = cfg
    
        self.bkbone = ResNet50()

        #high-level features
        self.aspp5 = ASPP_246(2048, 64)
        self.aspp4 = ASPP_246(1024, 64)
        self.H45 = HIF()

        # low-level features
        self.squeeze3 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.squeeze3d = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2d = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1d = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder = DualDecoder()

        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_88 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_80 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_72 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_64 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_56 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # edge branch
        self.lineare1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape

        out1h, out2h, out3h, out4h, out5v = self.bkbone(x)
        # high-level features
        out4, out5=self.aspp4(out4h), self.aspp5(out5v)
        out_h= self.H45(out4, out5)

        # low-level features for edge branch
        out1d = self.squeeze1d(out1h) 
        out2d = self.squeeze2d(out2h) 
        out3d = self.squeeze3d(out3h)
        # low-level features for saliency branch
        out1h, out2h, out3h = self.squeeze1(out1h), self.squeeze2(out2h), self.squeeze3(out3h)

        pred1, out1h,out2h, out3h,out1d,out2d,out3d= self.decoder(out1h, out2h, out3h,out1d, out2d, out3d, out_h)

        pred2 = F.interpolate(self.linearr1(out1h), size=shape, mode='bilinear', align_corners=True)
        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear', align_corners=True)
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear', align_corners=True)
        out4h = F.interpolate(self.linearr4(out_h), size=shape, mode='bilinear', align_corners=True)

        edge1 = F.interpolate(self.lineare1(out1d), size=shape, mode='bilinear', align_corners=True)
        edge2 = F.interpolate(self.lineare2(out2d), size=shape, mode='bilinear', align_corners=True)
        edge3 = F.interpolate(self.lineare3(out3d), size=shape, mode='bilinear', align_corners=True)

        # gated multi-scale (GMS) module
        if self.cfg.mode == 'train':
            if pred1.shape[2] == 88:
                pred1 = F.interpolate(self.linearr1_88(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 80:
                pred1 = F.interpolate(self.linearr1_80(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 72:
                pred1 = F.interpolate(self.linearr1_72(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 64:
                pred1 = F.interpolate(self.linearr1_64(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 56:
                pred1 = F.interpolate(self.linearr1_56(pred1), size=shape, mode='bilinear')
          
        else:
            pred_88 = F.interpolate(pred1, size=[88, 88], mode='bilinear')
            pred_88 = F.interpolate(self.linearr1_88(pred_88), size=shape, mode='bilinear')
            pred_80 = F.interpolate(pred1, size=[80, 80], mode='bilinear')
            pred_80 = F.interpolate(self.linearr1_80(pred_80), size=shape, mode='bilinear')
            pred_72 = F.interpolate(pred1, size=[72, 72], mode='bilinear')
            pred_72 = F.interpolate(self.linearr1_72(pred_72), size=shape, mode='bilinear')
            pred_64 = F.interpolate(pred1, size=[64, 64], mode='bilinear')
            pred_64 = F.interpolate(self.linearr1_64(pred_64), size=shape, mode='bilinear')
            pred_56 = F.interpolate(pred1, size=[56, 56], mode='bilinear')
            pred_56 = F.interpolate(self.linearr1_56(pred_56), size=shape, mode='bilinear')
            pred1 = 0.25*pred_88 + 1*pred_80 + 0.25*pred_72 + 0.25*pred_64 + 0.25*pred_56 
            
        return pred1, pred2, out2h, out3h, out4h, edge1, edge2, edge3

