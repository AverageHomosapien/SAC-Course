import gym
import numpy as np
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=4):
        super(NeuralNet, self).__init__()

        # Input to hidden layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1)
        return out

EPISODES = 100

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    Net = NeuralNet(observation_space, output_size = action_space)
    for i in range(EPISODES):
        state = env.reset()
        score = 0
        done = False
        while not done:
            state_, reward, done, info = env.step(np.random.choice(action_space))
            state = state_
            score += reward
        print(score)
