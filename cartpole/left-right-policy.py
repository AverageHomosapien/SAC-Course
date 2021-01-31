import numpy as np
import gym

BINS = 51
EPISODES = 5000

## Policy handles continuous state-spaces
## Focuses on the prediction problem well

class Agent():
    def __init__(self, alpha=0.1, gamma=0.99):
        self.V = {}
        self.min_max_pos = [-2.4, 2.4] # starts -0.05 to 0.05
        # self.min_max_degrees = [-12, 12]

        self.state_space = np.linspace(self.min_max_pos[0], self.min_max_pos[1], num=BINS)
        self.action_space = [0, 1]

        self.alpha = alpha
        self.gamma = gamma
        self.init_vals()

    def init_vals(self):
        for i in range(BINS):
            self.V[i] = 0.5

    def select_action(self, observation):
        bin = np.digitize(observation[0], self.state_space)
        return self.policy(bin)

    def policy(self, bin):
        if bin >= BINS/2:
            return 0
        return 1

    def update_V(self, obs, reward, obs_):
        obs_bin = np.digitize(obs[0], self.state_space)
        obs_bin_ = np.digitize(obs_[0], self.state_space)
        self.V[obs_bin] = self.V[obs_bin] + self.alpha * (reward + \
                      self.gamma * self.V[obs_bin_] - self.V[obs_bin])

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent()
    for i in range(EPISODES):
        observation = env.reset() # observation tuple (position, velocity, pole angle, pole velocity)
        done = False
        if i % (EPISODES//10) == 0:
            print("Episode " + str(i))
        while not done:
            action = agent.select_action(observation)
            observation_, reward, done, _ = env.step(action)
            agent.update_V(observation, reward, observation_)
            observation = observation_

    for i in range(BINS):
        print("Agent V for {} is {}".format(i, agent.V[i]))
