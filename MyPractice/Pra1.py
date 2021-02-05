import numpy as np
import torch

a = np.array([[1, 2], [3, 5]], dtype=float)
tensor_a = torch.tensor(a)
tensor_b = torch.from_numpy(a)
print('a', a)
print('tensor_a', tensor_a)
print('tensor_b', tensor_b)

a[1, 1] = 4.8
print('a', a)
print('tensor_a', tensor_a)
print('tensor_b', tensor_b)

tensor_c = torch.zeros_like(tensor_a, dtype=torch.int8)
print('tensor_c', tensor_c)
print(tensor_c.requires_grad)

out_d = torch.tensor([1])
print('out_d', out_d)
tensor_d = torch.zeros([2, 4], out=out_d)
print('tensor_d', tensor_d)
print('out_d', out_d)
