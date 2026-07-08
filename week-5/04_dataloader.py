import torch
from torch.utils.data import Dataset, DataLoader
import random

torch.manual_seed(42)
random.seed(42)


class LinearDataset(Dataset):
    def __init__(self, n_samples=5000):
        self.x = torch.randn(n_samples, 2)
        noise = 0.2 * torch.randn(n_samples, 1)
        self.y = (
            4 * self.x[:, 0:1]
            - 2 * self.x[:, 1:2]
            + 3
            + noise
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = LinearDataset()

print("Dataset Size:", len(dataset))

sample_x, sample_y = dataset[0]

print("\nFirst Sample")
print("Features:", sample_x)
print("Target:", sample_y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("\nTraining Batches:", len(train_loader))
print("Testing Batches:", len(test_loader))

for batch_idx, (features, targets) in enumerate(train_loader):
    print(f"\nBatch {batch_idx + 1}")
    print("Features Shape:", features.shape)
    print("Targets Shape:", targets.shape)

    if batch_idx == 2:
        break

feature_mean = torch.zeros(2)
feature_std = torch.zeros(2)

for features, _ in train_loader:
    feature_mean += features.mean(dim=0)
    feature_std += features.std(dim=0)

feature_mean /= len(train_loader)
feature_std /= len(train_loader)

print("\nFeature Mean:")
print(feature_mean)

print("\nFeature Standard Deviation:")
print(feature_std)

target_mean = 0
count = 0

for _, targets in train_loader:
    target_mean += targets.mean()
    count += 1

target_mean /= count

print("\nAverage Target Value:")
print(target_mean)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nUsing Device:", device)

for features, targets in train_loader:
    features = features.to(device)
    targets = targets.to(device)

    print("\nBatch on Device")
    print(features.device)
    print(targets.device)
    break
