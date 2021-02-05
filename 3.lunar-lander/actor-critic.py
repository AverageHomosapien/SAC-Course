import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, action_space, observation_space):
        super(ActorCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(observation_space, 2048)
        self.fc2 = nn.Linear(2048, 1536)
        self.fc3 = nn.Linear(1536, action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self):
        x = F.relu(fc1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCritic():
    def __init__(self, lr, gamma, action_space, observation_space):
        self.policy = ActorCriticNetwork(lr, action_space, observation_space)
        self.reward_memory = []
        self.action_memory = []
        self.scores = []

    # Selecting agent action
    def select_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probs = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    # Storing agent state rewards
    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    # Training the network
    def learn(self):
        self.policy.optimizer.zero_grad() # zero the gradient pre-learning

        # critic loss is delta ^2
        # actor loss is

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

        # how to get multiple outputs
        # how to update network based on actor and critic output
        # what v(s) is (not Gt)?
        # do we need additional memory to store through states?
        # do we need to change the output?


EPISODES = 3000

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = ActorCritic(0.00005, 0.99, env.action_space.n, env.observation_space.shape[0])

    for i in range(EPISODES):
        observation = env.reset()
        done = False
        while not done:
            action = agent.select_action(observation)
            observation_, reward, done, _ = env.step(action)
            agent.store_rewards(reward)
            agent.learn()
            observation = observation_
