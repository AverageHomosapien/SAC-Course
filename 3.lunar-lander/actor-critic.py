import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

## ACTOR CRITIC NETWORK TAKEN FROM PHIL - UPDATING ACTOR CRITIC MYSELF
class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions) # get actions (as normal) - ACTOR
        self.v = nn.Linear(fc2_dims, 1) # get the state value - CRITIC
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v) # return both outputs from network - with different layers


class ActorCritic():
    def __init__(self, lr, gamma, action_space, observation_space):
        self.policy = ActorCriticNetwork(lr, action_space, observation_space)
        self.reward_memory = []
        self.action_memory = []
        self.observation_memory = []
        self.gamma = gamma

    # Selecting agent action
    def select_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state[0]))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    # Storing agent state rewards
    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    # Storing current and next observation
    def store_observations(self, obs_tuple):
        self.observation_memory.append(obs_tuple)

    # Training the network -- my attempt (using additional list to store states and states_)
    ## don't check for last state - to zero (since no future rewards)
    def learn(self):
        self.policy.optimizer.zero_grad() # zero the gradient pre-learning

        loss = 0
        for idx, state in enumerate(self.observation_memory):
            Vs_t = self.policy.forward([state[0]])[1]
            Vs_t1 = self.policy.forward([state[1]])[1]
            delta = self.reward_memory[idx] + self.gamma * Vs_t1 - Vs_t
            loss += -delta * self.action_memory[idx]
        loss.backward()
        self.policy.optimizer.step()

        self.reward_memory = []
        self.action_memory = []
        self.observation_memory = []

        # how to get multiple outputs - create (n_actions output actor, 1 output critic) layers
        # do we need to change the output? - return both (just self.layer(x) for each)
        # how to update network based on actor and critic output
        # what v(s) is (not Gt)?
        # do we need additional memory to store through states?


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
            agent.store_observations((observation, observation_))
            agent.store_rewards(reward)
            agent.learn()
            observation = observation_
