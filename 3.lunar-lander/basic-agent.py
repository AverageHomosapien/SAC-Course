import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributions.distribution.Distribution as Distribution
import torch.distributions.categorical.Categorical as Categorical

EPISODES = 500

# REINFORCE Network
class PolicyNetwork(nn.Module):
    def __init__(self, lr, state_space, n_actions):
        super(PolicyNetwork, self).__init__()

        fc1 = nn.Linear(*state_space, 128)
        fc2 = nn.Linear(128, 128)
        fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self):
        x = F.relu(fc1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Agent class
class PolicyGradientAgent():
    def __init__(state_space, n_actions, gamma, lr=0.99):
        self.brain = PolicyNetwork(lr=lr, state_space = state_space, n_action = n_actions)
        self.gamma = gamma
        self.state_space = state_space
        self.action_space = n_actions
        self.rewards = []
        self.log_probs = []

    def select_action(self, state):
        probs = self.brain(state)
        m = Categorical(probs)
        action = m.sample()
        return action


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = PolicyGradientAgent(env.observation_space.shape[0], env.action_space.n, 0.99)
    total_score = []
    for i in range(EPISODES):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action()
            observation_, reward, done, info = env.step()
            agent.learn((observation, action, reward, observation_))
            observation = observation_
            score += reward
        total_score.append(score)
