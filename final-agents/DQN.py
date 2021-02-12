import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ReplayBuffer, plot_learning_curve

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.q(x)
        return actions

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                max_mem_size=1000000, eps_end = 0.01, eps_dec=5e-4,
                fc1_dims=256, fc2_dims=256):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.memory = ReplayBuffer(max_mem_size, input_dims, n_actions)
        self.network = DeepQNetwork(self.lr, input_dims, fc1_dims, fc2_dims, n_actions)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            return np.random.choice(self.action_space)
        state = T.tensor([observation]).to(self.network.device)
        actions = self.network.forward(state)
        action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.network.optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        states = T.tensor(states, dtype=T.float).to(self.network.device)
        #actions = T.tensor(actions, dtype=T.float).to(self.network.device)
        actions = [i[0] for i in actions]
        actions = np.array(actions)
        rewards = T.tensor(rewards, dtype=T.float).to(self.network.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.network.device)
        dones = T.tensor(dones).to(self.network.device)

        q_eval = self.network.forward(states)[batch_index, actions]
        q_next = self.network.forward(states_)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.network.loss(q_target, q_eval).to(self.network.device)
        loss.backward()
        self.network.optimizer.step()

        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_min)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                eps_end=0.01, input_dims=[8], lr=0.03) # cartpole n_actions=2, input_dims=[4]
    scores, eps_history = [], []
    n_games = 1000

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode: {}, score: {}, avg score: {}, eps: {}'.format(i, score, avg_score, agent.epsilon))

    x = [i+1 for i in range(n_games)]
    filename = 'CartPole_' + str(n_games) + '.png'
    plot_learning_curve(x, scores, filename)
