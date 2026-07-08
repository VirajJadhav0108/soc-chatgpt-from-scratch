import torch

torch.manual_seed(42)

scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

tensor3d = torch.rand(2, 3, 4)

print("Scalar:")
print(scalar)

print("\nVector:")
print(vector)

print("\nMatrix:")
print(matrix)

print("\n3D Tensor:")
print(tensor3d)

print("\nShape:", tensor3d.shape)
print("Dimensions:", tensor3d.ndim)
print("Data Type:", tensor3d.dtype)

zeros = torch.zeros((3, 3))
ones = torch.ones((3, 3))
identity = torch.eye(4)

print("\nZeros:")
print(zeros)

print("\nOnes:")
print(ones)

print("\nIdentity:")
print(identity)

random_tensor = torch.rand((3, 3))

print("\nRandom Tensor:")
print(random_tensor)

print("\nAddition:")
print(random_tensor + ones)

print("\nSubtraction:")
print(random_tensor - ones)

print("\nElement-wise Multiplication:")
print(random_tensor * ones)

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

print("\nMatrix Multiplication:")
print(torch.matmul(a, b))

print("\nTranspose:")
print(a.T)

reshaped = tensor3d.reshape(4, 6)
flattened = tensor3d.flatten()

print("\nReshaped Tensor:")
print(reshaped)

print("\nFlattened Tensor:")
print(flattened)

print("\nMean:", tensor3d.mean())
print("Max:", tensor3d.max())
print("Min:", tensor3d.min())
print("Sum:", tensor3d.sum())

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.rand((2, 2)).to(device)

print("\nDevice:", device)
print(tensor)

cpu_tensor = tensor.cpu()

print("\nMoved Back to CPU:")
print(cpu_tensor)
