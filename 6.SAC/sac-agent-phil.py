import gym
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.functional as F
from torch.distributions import Normal
from utils import ReplayBuffer, plot_learning_curve

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

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                name, chkpt_dir='tmp/sac'): # beta?
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = T.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = T.relu(action_value)
        action_value = self.q1(action_value)
        return action_value # no final layer activation

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action, n_actions,
                name, chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.reparam_noise = 1e-6 # noise for the re-parameterisation trick

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1) # test reparam noise
        return mu, sigma # no final layer activation

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.Tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise) # adding reparam noise incase log is 0
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs # return mu instead of action for deterministic

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, name, chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state_value)
        state_value = T.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = T.relu(state_value)
        state_value = self.fc3(state_value) # no final layer activation
        return action

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, alpha, beta, tau, env, env_id, gamma=0.99, n_actions=2,
                max_size=1e6, layer1_size=256, layer2_size=256, batch_size=100,
                reward_scale=2):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_action

        # can use shared critic input layer and different outputs
        # but in this case using 2 seperate critics
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                env.action_space.max_action, n_actions, env_id+'_actor')
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                n_actions, env_id + '_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                n_actions, env_id + '_critic_2')
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, env_id+'_value')
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, '_target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0] # returned as arr of arrays on gpu as torch tensor

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # overwriting parameters - setting new values
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print("... saving models")
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        print("... loading models")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self)
