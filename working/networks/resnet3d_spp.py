import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(1, '/trinity/home/xwan/MPNST_DL/working')
from networks.spp import SPP3DLayer

def get_inplanes(init = 64):
    init = init
    return [init, init * 2, init * 4, init * 8]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual =self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet_spp(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 scales,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.conv1 = nn.Conv3d(n_input_channels, 
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0])
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], stride=2)
        self.spp = SPP3DLayer(scales)
        self.linear = nn.Linear(block_inplanes[3] * sum([x**3 for x in scales]), n_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range (1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spp(x)
        x = self.linear(x)

        return x
    
def generate_model(model_depth, init, in_channel, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet_spp(BasicBlock, [1, 1, 1, 1], get_inplanes(init), scales=[5, 3, 1], n_input_channels=in_channel, **kwargs)
    elif model_depth == 18:
        model = ResNet_spp(BasicBlock, [2, 2, 2, 2], get_inplanes(init), scales=[5, 3, 1], n_input_channels=in_channel, **kwargs)
    elif model_depth == 34:
        model = ResNet_spp(BasicBlock, [3, 4, 6, 3], get_inplanes(init), scales=[5, 3, 1], n_input_channels=in_channel, **kwargs)
    elif model_depth == 50:
        model = ResNet_spp(Bottleneck, [3, 4, 6, 3], get_inplanes(init), scales=[5, 3, 1], n_input_channels=in_channel, **kwargs)
    elif model_depth == 101:
        model = ResNet_spp(Bottleneck, [3, 4, 23, 3], get_inplanes(init), scales=[5, 3, 1], n_input_channels=in_channel, **kwargs)
    elif model_depth == 152:
        model = ResNet_spp(Bottleneck, [3, 8, 36, 3], get_inplanes(init), scales=[5, 3, 1], n_input_channels=in_channel, **kwargs)
    elif model_depth == 200:
        model = ResNet_spp(Bottleneck, [3, 24, 36, 3], get_inplanes(init), scales=[5, 3, 1], n_input_channels=in_channel, **kwargs)

    return model

        
        
    
