# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn


from models.multiperception.anisotropic import Anisotropic_Attention
from models.multiperception.cross_task import Cross,PosCNN

import torch


class MultiperceptionCounting(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.feat_channels = in_channels

        self.relu = nn.ReLU(inplace=True)

        self.Csw_block1 = Anisotropic_Attention(self.feat_channels,16 ,1)
        self.Csw_block2 = Anisotropic_Attention(self.feat_channels, 16, 1)

        self.pos_ca_count=PosCNN(self.feat_channels,self.feat_channels)
        self.pos_ca_detect=PosCNN(self.feat_channels,self.feat_channels)

        self.count_convs = nn.ModuleList()
        self.detect_convs = nn.ModuleList()
        self.count_convs_2 = nn.ModuleList()
        self.detect_convs_2 = nn.ModuleList()

        self.count_convs.append(
            Cross(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )
        self.count_convs_2.append(
            Cross(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )
        self.detect_convs.append(
            Cross(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )
        self.detect_convs_2.append(
            Cross(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )



    def forward(self, x):


        decouple_x=self.Csw_block1(x)
        decouple_x = self.Csw_block2(decouple_x)
        count_feat, detect_feat=decouple_x,decouple_x
        count_feat=self.pos_ca_count(count_feat)
        detect_feat=self.pos_ca_detect(detect_feat)
        count_feat_=None
        detect_feat_=None


        for count_conv in self.count_convs:
            count_feat_ = count_conv(count_feat,detect_feat)
        for detect_conv in self.detect_convs:
            detect_feat_ = detect_conv(detect_feat,count_feat)

        for count_conv in self.count_convs_2:
            count_feat = count_conv(count_feat_,detect_feat_)
        for detect_conv in self.detect_convs_2:
            detect_feat = detect_conv(detect_feat_,count_feat_)


        return count_feat

if __name__=="__main__":
    dat = MultiperceptionCounting(256)
    fea = torch.randn(3, 256, 16, 16)
    f1, f2 = dat(fea)
    print()