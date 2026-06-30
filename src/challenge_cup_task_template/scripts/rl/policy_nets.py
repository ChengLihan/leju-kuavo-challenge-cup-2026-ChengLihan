#!/usr/bin/env python3
"""
Actor-Critic for deployment-realistic Kuavo grasping RL.

Single policy: 23-dim obs → 7-dim residual correction (±5°).
No gripper head — gripper is scripted (85% close, force feedback unavailable).

Designed for PPO via stable-baselines3 or custom training loop.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

MAX_RESIDUAL_RAD = np.deg2rad(5.0)


class ResidualActor(nn.Module):
    """23-dim obs → MLP → 7-dim residual (±5° rad)."""

    def __init__(self, obs_dim=23, action_dim=7, hidden=(256,256,128)):
        super().__init__()
        layers = []
        d_in = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(d_in, h), nn.ReLU()])
            d_in = h
        self.features = nn.Sequential(*layers)
        self.mean_head = nn.Linear(d_in, action_dim)
        # std ≈ exp(-3) ≈ 0.05 rad (~3°) — tight exploration around FK
        self.log_std = nn.Parameter(torch.ones(action_dim) * -3.0)

    def forward(self, obs):
        x = self.features(obs)  # (batch, d_in) → (batch, h[-1])
        mean = torch.tanh(self.mean_head(x)) * MAX_RESIDUAL_RAD  # (batch, 7)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs, action):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """23-dim obs → MLP → 1-dim value."""

    def __init__(self, obs_dim=23, hidden=(256,256,128)):
        super().__init__()
        layers = []
        d_in = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(d_in, h), nn.ReLU()])
            d_in = h
        self.features = nn.Sequential(*layers)
        self.value_head = nn.Linear(d_in, 1)

    def forward(self, obs):
        return self.value_head(self.features(obs)).squeeze(-1)
