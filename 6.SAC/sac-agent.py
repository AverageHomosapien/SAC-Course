import pybullet_envs
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


    def choose_action(self, state):
        state = T.tensor([observation]).to(self.actor.device)

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

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)

        # passing states and new states through value and target value networks
        # collapsing along batch dimension since we don't need 2d tensor for scalar quantities
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0 # setting terminal states to 0

        # pass current states through current policy get action & log prob values
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        # critic values for current policy state action pairs
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        # take critic min and collapse
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # actor loss (using reparam trick)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        # take critic min for new policy and collapse
        q1_new_policy = self.critic_1.forward(state, action)
        q2_new_policy = self.critic_2.forward(state, action)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # calculating actor loss
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        q_hat = self.scale * reward + self.gamma*value_ # qhat
        q1_old_policy = self.critic_1.forward(state, action).view(-1) # old policy (from replay buffer)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

# environments with large negative rewards don't work
# pybullet package looks good

if __name__ == '__main__':
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)
    agent = Agent(alpha=0.003, beta=0.003, reward_scale=2, env_id=env_id,
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=256, layer1_size=256, layer2_size=256,
                n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = env_id+'_'+str(n_games)+'games_scale'+str(agent.scale)+'.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode {} score {} trailing 100 games avg {} steps {} env {}'.format(
            episode, score, avg_score, steps, env_id))
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
