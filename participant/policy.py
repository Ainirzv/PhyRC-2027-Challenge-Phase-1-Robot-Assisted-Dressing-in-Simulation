
"""Behavior-cloning policy for the PhyRC challenge."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BehaviorCloningPolicy:
    def __init__(self, checkpoint, action_space):
        self.device = torch.device("cpu")

        # Load the trained checkpoint
        saved = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        self.mean = np.asarray(saved["state_mean"], dtype=np.float32)
        self.std = np.asarray(saved["state_std"], dtype=np.float32)

        input_dim = int(saved["input_dim"])

        # This architecture must match train_bc.py
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 18),
            nn.Tanh(),
        )

        self.model.load_state_dict(saved["model_state_dict"])
        self.model.eval()

        self.action_space = action_space

    def reset(self):
        pass

    def __call__(self, observation):
        # Use the same feature order as training
        keys = ["joint_position", "joint_velocity", "previous_action"]

        state = np.concatenate([
            np.asarray(observation[k], dtype=np.float32).ravel()
            for k in keys
        ])

        state = (state - self.mean) / self.std

        with torch.no_grad():
            tensor = torch.from_numpy(state.astype(np.float32))
            action = self.model(tensor).cpu().numpy()

        action = action.reshape(2, 9).astype(np.float32)

        # Keep actions within the allowed range
        action = np.clip(action, -1.0, 1.0)

        return action


def load_policy(checkpoint, observation_space, action_space):
    if checkpoint is None:
        raise ValueError("A trained checkpoint must be provided.")

    return BehaviorCloningPolicy(checkpoint, action_space)

