
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Find the project directory automatically
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src" / "PhyRC_Sim"))
from Policy.dataset_reader import PolicyDataset

# Load the recorded demonstration
data_path = ROOT / "output/policy_datasets/20261020T055633_377126Z_dc1b2c/policy.hdf5"
save_path = ROOT / "output/checkpoints/policy.pt"

keys = ["joint_position", "joint_velocity", "previous_action"]

print("Loading dataset:", data_path)
dataset = PolicyDataset(str(data_path), observation_keys=keys)

X, Y = [], []

for i in range(len(dataset)):
    obs, action = dataset[i]
    state = np.concatenate([obs[k].ravel() for k in keys])
    X.append(state)
    Y.append(np.asarray(action).ravel())

dataset.close()

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

print("Training samples:", len(X))
print("Input shape:", X.shape)
print("Action shape:", Y.shape)

# Check that actions contain the expected 18 values
if Y.shape[1] != 18:
    raise ValueError(f"Expected 18 action values, got {Y.shape[1]}")

# Normalize state inputs
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std < 1e-6] = 1.0
X = (X - mean) / std

torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(X.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 18),
    nn.Tanh()
)

loader = DataLoader(
    TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
    batch_size=64,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Initial pipeline test: train on the recorded samples
for epoch in range(20):
    total_loss = 0.0

    for states, actions in loader:
        prediction = model(states)
        loss = loss_fn(prediction, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: loss = {total_loss / len(loader):.5f}")

# Save model weights and normalization values
save_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "state_mean": mean,
    "state_std": std,
    "input_dim": X.shape[1],
}, save_path)

print("Model saved to:", save_path)

