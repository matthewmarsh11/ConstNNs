import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Generate dummy data
np.random.seed(0)
x = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(*x.shape)

x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 2. Define simple NN model
class QuantileRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# 3. Define quantile loss (pinball loss)
def quantile_loss(y_pred, y_true, tau):
    diff = y_true - y_pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))

# 4. Train model for each quantile
def train_model(tau, x_train, y_train, epochs=2000):
    model = QuantileRegressor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = quantile_loss(y_pred, y_train, tau)
        loss.backward()
        optimizer.step()
    return model

# 5. Train models
taus = [0.1, 0.5, 0.9]
models = [train_model(tau, x_tensor, y_tensor) for tau in taus]

# 6. Predict and plot
x_test = torch.linspace(0, 1, 200).unsqueeze(1)
with torch.no_grad():
    preds = [model(x_test).squeeze().numpy() for model in models]

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', alpha=0.5, label='Data')
colors = ['blue', 'green', 'red']
for i, tau in enumerate(taus):
    plt.plot(x_test.squeeze(), preds[i], label=f'Quantile {tau}', color=colors[i])
# plt.title('Quantile Regression with Pinball Loss\n(Demonstrating Quantile Crossing)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()