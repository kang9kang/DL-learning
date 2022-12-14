import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.float64)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

print(x.size())
print(x.shape)

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

y = x[0, :]
y += 1
print(y)
print(x[0, :])

y = x.view(15)
z = x.view(-1, 5)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x += 1
print(x)
print(y)

x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

x = torch.randn(1)
print(x)
print(x.item())

x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before)

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y)
print(id(y) == id_before)

a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

c = torch.tensor(a)
a += 1
print(a, c)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
