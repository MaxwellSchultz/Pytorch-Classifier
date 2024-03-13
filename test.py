import numpy as np
from scipy.sparse import csr_matrix
import torch

features_file = "task1_topics/train.sparseX"
targets_file = "task1_topics/train.CT"

features = np.loadtxt(features_file, dtype=np.int64)
targets = np.loadtxt(targets_file, dtype=np.int64)

features = torch.tensor(features)
targets = torch.tensor(targets)
print(features)
# print(targets)

# Convert sparse data into indices, values
indices = torch.tensor([[entry[0], entry[1]] for entry in features], dtype=torch.long).t()
values = torch.tensor([entry[2] for entry in features], dtype=torch.float)

# Create a sparse tensor
sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size([113295, 100000]))

# Convert sparse tensor to dense tensor
dense_tensor = sparse_tensor.to_dense()

print("Sparse Tensor:")
print(sparse_tensor)
print("\nDense Tensor:")
print(dense_tensor)
print("0 0", dense_tensor[0][0]) #1
print("0 1", dense_tensor[0][1]) #1
print("0 101", dense_tensor[0][101]) #1
print("52726 9828", dense_tensor[52726][9828]) #2
print("52737 65", dense_tensor[52737][65]) #3
print("52741 312", dense_tensor[52741][312]) #1




