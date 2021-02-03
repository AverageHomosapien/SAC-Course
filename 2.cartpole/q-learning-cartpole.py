import numpy as np
import gym
import matplotlib.pyplot as plt

# digitize
# Q(s,a) instead of v(s)

class Agent():
    def __init__(self, alpha=0.1, gamma=0.9, eps=1.0, eps_min=0.04, eps_dec=0.99997):
        self.memory = []
        self.bins_cart_pos = np.linspace(-3, 3, num=BINS)
        self.bins_cart_vel = np.linspace(-4, 4, num=BINS)
        self.bins_pole_angle = np.linspace(-0.3, 0.3, num=BINS)
        self.bins_pole_velocity = np.linspace(-4, 4, num=BINS)

        self.state_space = {}
        self.action_space = [0, 1]
        self.Q = {}

        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.alpha = alpha
        self.gamma = gamma
        self.init_vals()

    def init_vals(self):
        for i in range(BINS):
            for j in range(BINS):
                for k in range(BINS):
                    for l in range(BINS):
                        state = (i, j, k, l)
                        for action in self.action_space:
                            self.Q[(state, action)] = 0

    def policy(self, state):
        if np.random.rand() < self.eps: # select non-random action
            digitized_state = self.digitize_state(state)
            q_vals = []
            for action in self.action_space:
                q_vals.append(self.Q[digitized_state, action])
            self.eps *= self.eps_dec
            self.eps = max(self.eps, self.eps_min)
            return np.argmax(q_vals)
        return np.random.choice(self.action_space) # select random action

    def update_Q(self, sars_):
        state, action, reward, state_ = sars_
        digitized_state = self.digitize_state(state)
        digitized_next = self.digitize_state(state_)
        q_vals = []
        for action in self.action_space:
            q_vals.append(self.Q[digitized_next, action])
        self.Q[digitized_state, action] = self.Q[digitized_state, action] + self.alpha * \
                            (reward + self.gamma * max(q_vals) - self.Q[digitized_state, action])

    def digitize_state(self, state):
        return (np.digitize(state[0], self.bins_cart_pos), np.digitize(state[1], self.bins_cart_vel),\
                   np.digitize(state[2], self.bins_pole_angle), np.digitize(state[3], self.bins_pole_velocity))

EPISODES = 50000
BINS = 10
SPLITS = 100

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = Agent()
    rewards = []
    for i in range(EPISODES):
        if i % (EPISODES//SPLITS) == 0:
            print("Done {} episodes, eps is {}".format(i, agent.eps))
            rewards.append(0)
        observation = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info = env.step(action)
            agent.update_Q((observation, action, reward, observation_))
            observation = observation_
            rewards[-1] += 1

    for i in range(len(rewards)):
        rewards[i] /= (EPISODES//SPLITS)
    x = np.arange(SPLITS)
    y = rewards
    plt.title("Average episode reward")
    plt.plot(x,y)
    plt.show()

    for action in agent.action_space:
        print("agent q is {} for action {} and state {}".format(agent.Q[((8,8,8,8), action)], action, (8,8,8,8)))
        print("agent q is {} for action {} and state {}".format(agent.Q[((7,7,7,7), action)], action, (7,7,7,7)))
        print("agent q is {} for action {} and state {}".format(agent.Q[((6,6,6,6), action)], action, (6,6,6,6)))
        print("agent q is {} for action {} and state {}".format(agent.Q[((5,5,5,5), action)], action, (5,5,5,5)))
        print("agent q is {} for action {} and state {}".format(agent.Q[((4,4,4,4), action)], action, (4,4,4,4)))
        print("agent q is {} for action {} and state {}".format(agent.Q[((3,3,3,3), action)], action, (3,3,3,3)))
