import torch
from torch import nn
print(torch.cuda.is_available()) # 输出 True
print(torch.cuda.device_count()) # 输出 1
print(torch.cuda.current_device()) # 输出 0
print(torch.cuda.get_device_name(0)) # 输出 GeForce GTX 1080 Ti

x = torch.tensor([1, 2, 3])
print(x)
print(x.device)
x = x.cuda()
print(x)
print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
print(x)

y = x**2
print(y)
# z = y + x.cpu()
net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
net.cuda()
print(list(net.parameters())[0].device)

x = torch.rand(2,3).cuda()
print(net(x))