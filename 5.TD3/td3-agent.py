import gym
import os
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random.gauss as gauss
from utils import ReplayBuffer, plot_learning_curve

# Lots of the imp removed from the DDPG (normalization etc)

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims) # passing in actions and input_dimensions
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta) # no weight decay like in ddpg
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        s_a_value = self.fc1(T.cat([state, action], dim=1)) # concatenate state & action - not add
        s_a_value = F.relu(s_a_value)
        s_a_value = self.fc2(s_a_value)
        s_a_value = F.relu(s_a_value)
        s_a_value = self.q1(s_a_value) # no tanh on critic
        return s_a_value

    def save_checkpoint(self):
        print("..saving checkpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint")
        T.load_state_dict(T.load(self.checkpoint_file))


def ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # only takes state - critic takes state/action
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) # in paper says 300-1 layer (but can't predict actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        #x = F.tanh(self.mu(x))
        x = T.tanh(self.mu(x)) # if action is >+/- then multiply by max action

        return x

    def save_checkpoint(self):
        print("..saving checkpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint")
        T.load_state_dict(T.load(self.checkpoint_file))


class Agent(object):
    # needs functions init, choose_action, store_transition
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, alpha, beta,
                batch_size=100, max_size=1e6, mu=0, sigma=0.1):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta

        self.batch_size = batch_size
        self.max_size = max_size
        self.noise = gauss(mu, sigma)
        #self.clamp = max(0.5, x)?

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, 'actor_net')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'critic_net')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'target_critic')
        self.memory = ReplayBuffer(self.max_size, input_dims, n_actions, batch_size=self.batch_size)

    def choose_action(self, state):





if __name__ == '__main__':
