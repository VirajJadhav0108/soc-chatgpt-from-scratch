import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


x = torch.randn(1000, 2)
y = (3 * x[:, 0] - 2 * x[:, 1] + 5).unsqueeze(1)

model = FeedForwardNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    predictions = model(x)

    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1:03d} | Loss: {loss.item():.6f}")

test = torch.tensor([
    [2.0, 1.0],
    [4.0, -1.0],
    [-2.0, 3.0]
])

with torch.no_grad():
    outputs = model(test)

print("\nPredictions:")
print(outputs)

torch.save(model.state_dict(), "feedforward_model.pth")

loaded_model = FeedForwardNetwork()
loaded_model.load_state_dict(torch.load("feedforward_model.pth"))
loaded_model.eval()

with torch.no_grad():
    predictions = loaded_model(test)

print("\nLoaded Model Predictions:")
print(predictions)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nTotal Parameters:", total_params)
print("Trainable Parameters:", trainable_params)
