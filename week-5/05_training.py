import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(42)


class RegressionDataset(Dataset):
    def __init__(self, samples=10000):
        self.x = torch.randn(samples, 2)
        noise = 0.1 * torch.randn(samples, 1)
        self.y = (
            3 * self.x[:, 0:1]
            - 2 * self.x[:, 1:2]
            + 5
            + noise
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RegressionDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
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

model = RegressionModel().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 50

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for features, targets in train_loader:

        features = features.to(device)
        targets = targets.to(device)

        predictions = model(features)

        loss = criterion(predictions, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    model.eval()

    test_loss = 0

    with torch.no_grad():

        for features, targets in test_loader:

            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)

            loss = criterion(predictions, targets)

            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Train Loss: {average_loss:.6f} | "
        f"Test Loss: {average_test_loss:.6f}"
    )

torch.save(model.state_dict(), "regression_model.pth")

loaded_model = RegressionModel().to(device)

loaded_model.load_state_dict(
    torch.load("regression_model.pth", map_location=device)
)

loaded_model.eval()

sample = torch.tensor([
    [2.0, 1.0],
    [0.5, -1.0],
    [-2.0, 3.0]
]).to(device)

with torch.no_grad():
    predictions = loaded_model(sample)

print("\nPredictions:")
print(predictions.cpu())

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTotal Parameters:", total_parameters)
print("Trainable Parameters:", trainable_parameters)
