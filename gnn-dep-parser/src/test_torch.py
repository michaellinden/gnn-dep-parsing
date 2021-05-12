import torch

# (batch_size, 5, 10, 20)
# key.transpose(-2, -1)
query = torch.rand(100, 5, 10, 20)
key = query

print(query.size())
print(key.transpose(-2, -1).size())

res = torch.matmul(query, key.transpose(-2, -1))

print(res.size())



