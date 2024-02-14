import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# add
z1 = torch.empty(3)
torch.add(x, y, out = z1)

z2 = torch.add(x,y)
z = x+y

# sub
z = x-y

# div
z = torch.true_divide(x, y)

# inplace operation
t = torch.zeros(3)
t.add(x)
t += x


# exponentiation
z= x.pow(2)
z = x ** 2

# single comparison
z = x > 0
z = x < 0

# matrrix mul
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)

# matrix_exp
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# element wise multiplication
z = x * y

# dot product\
z = torch.dot(x, y)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)

# eg of broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
