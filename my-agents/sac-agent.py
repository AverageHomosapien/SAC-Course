import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.functional as F
# V, Q, Policy networks
# V takes states -> outputs scalar value
# Critic takes states & actions, outputs scalar value (incorp actions first layer and use concat)
# Actor takes states -> outputs mean and sigma
    # Constrain sigma with clamp (0,1) or sigmoid
# Pytorch normal distribution for actions
# Reparameterisation trick (sample vs rsample) - boolean for Reparameterisation?

# Log prob has dimensions n_actions, need to sum/mean
# Multivariate normal distribution doesn't seem to work
# Can also copy over the functionality to save networks
# Check appendix for hyperparameters

class ValueNetwork(nn.Module):
    def __init__(self, lr, gamma, input_dims, fc1_dims, fc2_dims, n_actions, tau):
        super(ValueNetwork, self).__init__()
        self.gamma = gamma
        self.tau = tau

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        v = self.fc1(state)
        v = F.relu(v)
        v = self.fc2(v)
        v = F.relu(v)
        v = self.fc3(v) # last layer? like tanh?
        return v

    def save_checkpoint(self):
        print("..saving checkpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint")
        T.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, lr, gamma, input_dims, fc1_dims, fc2_dims, n_actions): # alpha?
        super(ActorNetwork, self).__init__()
        self.gamma = gamma

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        action = self.fc1(state)
        action = T.relu(state)
        action = self.fc2(state)
        action = T.relu(state)
        action = self.fc3(state) # final layer?
        return action

    def save_checkpoint(self):
        print("..saving checkpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint")
        T.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, lr, gamma, input_dims, fc1_dims, fc2_dims, n_actions): # beta?
        super(CriticNetwork, self).__init__()
        self.gamma = gamma
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions


        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        p = self.fc1(T.cat([state, action], dim=1))
        p = T.relu(p)
        p = self.fc2(p)
        p = T.relu(p)
        return p

    def save_checkpoint(self):
        print("..saving checkpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint")
        T.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, tau, gamma, alpha, beta, fc1_dims, fc2_dims, input_dims,
                n_actions, env, max_size=1e6):

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                env.action_space.max_action, n_actions, 'actor_net')
        self.q1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'q1_net')
        self.q2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'q2_net')
        self.v = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, 'value_net')
        self.target_v = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, 'target_value_net')


    def choose_action():


    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        q1_params = self.q1.named_parameters()
        q2_params = self.q2.named_parameters()
        v_params = self.v.named_parameters()
        target_v_params = self.target_v.named_parameters()

        actor_state_dict = dict(actor_params)
        q1_state_dict = dict(q1_params)
        q2_state_dict = dict(q2_params)
        v_state_dict = dict(v_params)
        target_v_state_dict = dict(target_v_params)

        # overwriting parameters - setting new values
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*actor_state_dict[name].clone()

        for name in q1_state_dict:
            q1_state_dict[name] = tau*q1_state_dict[name].clone() + \
                (1-tau)*q1_state_dict[name].clone()

        for name in v_state_dict:
            v_state_dict[name] = tau*v_state_dict[name].clone() + \
                (1-tau)*v_state_dict[name].clone()

        self.q2.load_state_dict(q1_state_dict)
        self.target_v_state_dict.load_state_dict(v_state_dict)

    def save_model(self):
        self.actor.save_checkpoint()
        self.q1.save_checkpoint()
        self.q2.save_checkpoint()
        self.v.save_checkpoint()
        self.target_v.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.q1.load_checkpoint()
        self.q2.load_checkpoint()
        self.v.load_checkpoint()
        self.target_v.load_checkpoint()
