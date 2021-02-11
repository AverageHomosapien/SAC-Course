import gym
import os
import datetime
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
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
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    # needs functions init, choose_action, store_transition
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                update_actor_interval=2, n_actions = 2, warmup=1000, max_size=1e6,
                layer1_size=400, layer2_size=300, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        #self.max_action = n_actions
        #self.min_action = 0

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0 # how often to call the learning function on the actor network
        self.time_step = 0 # handles countdown to end of warmup
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions, 'actor_net')
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, 'critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, 'critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, name='target_critic_2')

        self.noise = noise
        self.update_network_parameters(tau=1) # sets the target network parameters to original

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions, ))).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        # clamping on action to make sure it stays in range
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)

        target_actions = self.actor.forward(state_) # get the new states
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5) # add noise
        #target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])
        # clamp to ensure target action in bounds of environment () - may break if -ve element != -(+ve) element

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        # needed for loss function
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        #handle when new states are terminal
        q1_[done] = 0.0
        q2_[done] = 0.0

        # collapse on batch dimension
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_) # perform minimisation operation (to get min)
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1) # add batch dimension to feed through loss function

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # Calculate and sum losses (can only backprop once in pytorch)
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward() # backprop

        # step optimizer
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr +=1
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state)) # actor loss proportional to loss of critic net (1)
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None): # using soft update rule
        # called at beginning of initializer to set init network params
        if tau is None:
            tau = self.tau

        # get the named parameters of every network
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        # converting to dicts
        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        # overwriting parameters - setting new values
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_model(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()


LOAD_EXISTING = False

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape,
                tau=0.005, env=env, batch_size=100, layer1_size=400, layer2_size=300,
                n_actions=env.action_space.shape[0]) # 2 actions for lunar lander
    n_games = 2000
    filename = 'LunarLander_' + str(n_games) + "_"
    figure_file = 'plots/' + filename + str(datetime.datetime.now().microsecond) + '.png'

    best_score = env.reward_range[0]
    score_history = []

    if LOAD_EXISTING:
        agent.load_models()
        print("... loading checkpoint")

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            print("... saving checkpoint")
            best_score = avg_score
            agent.save_model()
        print('episode {} score {} trailing 100 games avg {}'.format(i, round(score, 2), round(avg_score, 3)))

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
