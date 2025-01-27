import torch
from thop import profile
from models.ffmc import FFMC

model = FFMC()
model.cuda()



input1 = torch.randn(1, 3, 384, 576).cuda()
input2 = torch.randn(3, 5).cuda()
flops, params = profile(model, inputs=(input1, input2))
# sum1 = summary(model, [input1,input2])
# print(sum1)
total_parameters = sum(p.numel() for p in model.parameters())
total_parameters2 = params
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print((f"total number of params: {total_parameters / 1e6}M"))
print((f"total number of params2: {total_parameters2 / 1e6}M"))
print((f"number of params: {n_parameters / 1e6}M"))
print((f"number of GFLOPs: {flops / 1e9}GFLOPs"))