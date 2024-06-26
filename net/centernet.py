'''
2. centernet.py（网络总模块）
    1. centernet类（这里面就整合了三种网络，这三种子网络就不用改了，直接调用这里面的三者就行）
    2. 骨干（resnet）冻结函数
    3. 骨干解冻函数
    4. 初始化权重函数
'''

import math

import torch.nn as nn
from torch import nn

from net.resnet50 import resnet50, resnet50_Decoder, resnet50_Head

class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes = 20, pretrained = False):
        super(CenterNet_Resnet50, self).__init__()

        self.pretrained = pretrained

        # 512,512,3 -> 16,16,2048
        self.backbone = resnet50(pretrained = pretrained)

        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)

        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)
        
        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))
