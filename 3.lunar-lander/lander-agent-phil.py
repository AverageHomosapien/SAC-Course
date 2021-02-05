import numpy as np
import gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()

        # fully connected
        self.fc1 = nn.Linear(input_dims, 128) # * unpacks a list
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state): # output goes through forward
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x # either return x,y or save where softmax is used?

class PolicyGradientAgent():
    def __init__(self, lr, input_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = [] # to keep track of rewards in the episode
        self.action_memory = [] # to keep track of log probs of actions taken

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device) # change observation's dimensionality to pytorch batch
        # in here too https://pytorch.org/docs/stable/distributions.html
        probabilities = F.softmax(self.policy.forward(state)) # softmax ff output to sum outputs to 1
        action_probs = T.distributions.Categorical(probabilities) # getting action probabilities
        action = action_probs.sample() # sampling action - tensor([0]..)
        log_probs = action_probs.log_prob(action) # getting log probs of action - tensor([-1.2246]..)
        self.action_memory.append(log_probs)
        return action.item() # changing from pytorch tensor

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad() # zero the gradient pre-learning

        # Implement one or other:
        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
        G = np.zeros_like(self.reward_memory)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.Tensor(G).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        self.reward_memory = []
        self.action_memory = []


def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(0.0005, env.observation_space.shape[0], env.action_space.n)
    fname = 'REINFORCE_' + 'lunar_lander_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            observation = observation_
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)
