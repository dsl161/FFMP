import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d


class Backbone(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained: bool,
        dilation: bool,
        reduction: int,
        swav: bool,
        requires_grad: bool
    ):

        super(Backbone, self).__init__()

        resnet = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained = True
        )

        self.backbone = resnet
        self.reduction = reduction

        if name == 'resnet50' and swav:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
                map_location="cpu"
            )
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        # concatenation of layers 2, 3 and 4
        self.num_channels = 896 if name in ['resnet18', 'resnet34'] else 3584
        self.input_proj = nn.Conv2d(
            self.num_channels, 256, kernel_size=1
        )

        for n, param in self.backbone.named_parameters():
            if 'layer2' not in n and 'layer3' not in n and 'layer4' not in n:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)

        # self.clsfc = nn.Conv2d(3584, 256, 1, padding=0)
        # self.denfc = nn.Conv2d(3584, 256, 1, padding=0)

    def forward(self, x):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = layer2 = self.backbone.layer2(x)
        x = layer3 = self.backbone.layer3(x)
        x = layer4 = self.backbone.layer4(x)

        x = torch.cat([
            F.interpolate(f, size=size, mode='bilinear', align_corners=True)
            for f in [layer2, layer3, layer4]
        ], dim=1)

        x = self.input_proj(x) #(1,3584,64,64)

        # return self.clsfc(x), self.denfc(x)#(1,256,64,64)

if __name__=="__main__":
    backbone = Backbone(name = 'resnet50', pretrained=True, dilation=False, reduction=8,swav=True, requires_grad=False)
    a = torch.rand(1, 3, 384, 384)
    b, c = backbone(a)