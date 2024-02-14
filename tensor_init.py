# tensor init
import torch

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype = torch.float32, device="cuda")

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)


# init methods
x = torch.empty(size = (3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye((5))
x = torch.arange(start =0, end = 5, step = 1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size = (1, 5)).normal_(mean=0, std = 1)
x = torch.diag(torch.ones(3))

# init and covert
tensor = torch.arange(4)
# print true false
print(tensor.bool())
# int16
print(tensor.short())
# int64 imp
print(tensor.long())
# float16
print(tensor.half())
# float32
print(tensor.float())
# float64
print(tensor.double())


# array to tensor and vice versa

import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()