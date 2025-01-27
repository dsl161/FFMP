from typing import Tuple

from torch import Tensor

import torch
import torch.nn as nn
from models.group import GroupOp
class Topk(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.cross = GroupOp()

        self.emb = nn.Linear(qk_dim, qk_dim)
        self.emb2 = nn.Linear(qk_dim*2, qk_dim)

    def forward(self, clsfea, denfea, anchor_cls, bs_mean, b) -> Tuple[Tensor]:

        _,_,a,a1 = clsfea.shape
        a2 = 384 * 576 / a / a1

        size = int(bs_mean[0][0] * bs_mean[0][1] // a2)
        size = max(size, 3)
        mask = self.cross(clsfea, anchor_cls)
        sim = denfea * mask  # torch.Size([1, 256, 48, 48])
        top = sim.flatten(-2, -1)
        topk_attn_logit, topk_index = torch.topk(top, k=size, dim=-1)  # (n, p^2, k), (n, p^2, k)
        top1 = clsfea.flatten(-2, -1).gather(dim=2, index=topk_index)
        top1 = self.emb(top1.transpose(-1, -2))
        anchor_cls1 = top1.view(b, size, -1, 1, 1).mean(dim=1)
        anchor_cls2 = torch.cat((anchor_cls, anchor_cls1), dim=1)
        anchor_cls2 = self.emb2(anchor_cls2.transpose(-1, -3)).transpose(-1, -3)

        return anchor_cls2


if __name__=="__main__":
    top = Topk(256)
    a = torch.rand(3, 256, 28, 28)
    d = torch.rand(3, 256, 28, 28)
    e = torch.rand(3, 256, 1, 1)
    f = torch.tensor([[36, 51]])
    b = top(a, d, e, f, 3)
    print(b)
    # a= clsfea.shape[3]
    # b = 384*384/a/a
    #
    # size = int(bs_mean[0][0] * bs_mean[0][1] // b)
    # size = max(size, 3)