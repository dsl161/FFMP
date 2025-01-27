import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupOp(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, feature, anchor, mask=None, boxes=None):
        cos_sim = F.normalize(feature, 2, dim=1) * F.normalize(anchor, 2, dim=1)
        cos_sim = F.relu(cos_sim.sum(dim=1, keepdim=True), inplace=True)
        return cos_sim
