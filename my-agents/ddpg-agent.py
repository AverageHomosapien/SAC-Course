import gym
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils_phil import ReplayBuffer, OUActionNoise, plot_learning_curve
# Used Adam Optimizer
# Used LR (10e-4 for actor), (10e-3 for critic) - both target and copy
# For Q, L2 weight decay of 10e-2
# Gamma = 0.99
# Tau = 0.001
# Used ReLU for all hidden layers
# Used tanh layer for output
# Low dimensional was 2 hidden layers (400, 300 units)
# Minibatch size of 64
# Replay buffer size of 10e6
# for noise, (Ornstein-Uhlenbeck) - Theta = 0.15 and sigma = 0.2


# State is input from first hidden layer
# Action is input from second hidden layer
class Critic():
    def __init__(self, input_dims, lr=1e-3, l2_decay=0.01, fc1_dims=400, fc2_dims=300):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Tanh(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self):
        x = F.relu(self.fc1())
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def weights_init(self, m):
        if type(m) == nn.Linear:
            y = -1/sqrt()
        elif type(m) == nn.Tanh:



class Agent():


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
