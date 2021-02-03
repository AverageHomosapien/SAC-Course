import gym
import numpy as np
import torch.nn as nn

EPISODES = 20

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    Net = NeuralNet(observation_space, output_size = action_space)
    for i in range(EPISODES):
        state = env.reset()
        score = 0
        env.render()
        done = False
        while not done:
            state_, reward, done, info = env.step(np.random.choice(action_space))
            state = state_
            score += reward
            env.render()
        print(score)
