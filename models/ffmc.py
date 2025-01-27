# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.roialign import ROIAlign
from models.group import GroupOp
from models.topk import Topk
from models.frequency import FrequencyDomainLayer
from models.backbone.vgg19 import Vgg19
from models.regression import DensityRegressor

from models.multiperception.MultiperceptionCounting import MultiperceptionCounting


class FFMC(nn.Module):
    def __init__(self,  monemtum=False):
        super().__init__()
        self.encoder = Vgg19()
        feadim = self.encoder.outdim()
        self.roi = 16
        self.freq1 = FrequencyDomainLayer(feadim, 14, 14, freq_sel_method="top2")
        self.roialign = ROIAlign(self.roi, 1./8)
        self.cross = GroupOp()
        self.linear = nn.Linear(feadim, feadim, bias=False)
        self.decoder = DensityRegressor(feadim)
        self.multi = MultiperceptionCounting(256)
        self.top = Topk(feadim, topk=10)
        self.momentum = monemtum
    def forward(self, image, boxes):
        bsize = torch.stack((boxes[:, 4] - boxes[:, 2], boxes[:, 3] - boxes[:, 1]), dim=-1)
        bs_mean = bsize.view(-1, 3, 2).float().mean(dim=1)
        b, _, imh, imw = image.shape
        clsfea, denfea = self.encoder(image, bs_mean)
        clsfea = self.freq1(clsfea)
        patches = self.roialign(clsfea, boxes)
        anchors_patchs = patches.view(b, 3, -1, self.roi, self.roi).mean(dim=1)  # torch.Size([1, 256, 16, 16])
        anchor_cls = anchors_patchs.mean(dim=(-1, -2), keepdim=True)  # torch.Size([1, 256, 1, 1])
        anchor_cls2 = self.top(clsfea, denfea, anchor_cls, bs_mean, b)
        clsfea1 = self.multi(clsfea)
        mask = self.cross(clsfea1, anchor_cls2)  # torch.Size([1, 1, 48, 72])
        denmap = self.decoder(denfea * mask)



        return denmap

if __name__ == '__main__':
    vgg = FFMC()

    feature = torch.rand(2, 3, 384, 384)
    boxes = torch.tensor([
        [0.0000, 197.4755, 266.0000, 269.6972, 317.0000],
        [0.0000, 349.8337,  66.0000, 395.3433,  86.0000],
        [0.0000, 205.3902, 102.0000, 240.0171, 139.0000],
        [1.0000, 197.4755, 266.0000, 269.6972, 317.0000],
        [1.0000, 349.8337, 66.0000, 395.3433, 86.0000],
        [1.0000, 205.3902, 102.0000, 240.0171, 139.0000]
                          ])
    print(boxes.shape)
    den = vgg(feature,boxes)
    print(den.shape) #torch.Size([1, 1, 384, 384])
