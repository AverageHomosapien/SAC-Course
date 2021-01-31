import numpy as np
import random
import gym
import matplotlib.pyplot as plt

# dec of 0.99995 for 50,000
# dec of 0.999987 for 200,000

NO_EPISODES = 200000
RECORDINGS = 500

class Agent():
    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_dec = 0.999987, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

        self.Q = {} # estimate of state value
        self.sum_space = [i for i in range(4, 22)] # bust above 21 (not really part of sum space)
        self.dealer_show_card_space = [i+1 for i in range(10)]
        self.ace_space = [False, True]


        self.action_space = [0, 1] # stick or hit
        self.state_space = []

        self.returns = {}
        self.pairs_visited = {} # First visit Monte Carlo
        self.memory = []
        self.gamma = gamma

        self.init_vals()

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    state = (total, card, ace)
                    self.state_space.append(state)
                    for action in self.action_space:
                        self.Q[(state, action)] = 0
                        self.returns[(state, action)] = []
                        self.pairs_visited[(state, action)] = 0

    def policy(self, state):
        action = 0
        self.epsilon *= self.epsilon_dec
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if random.random() > self.epsilon:
            best = 0
            for idx, action in enumerate(self.action_space):
                if self.Q[(state, action)] > self.Q[(state, best)]:
                    best = idx
            return best
        return random.randint(self.action_space[0], self.action_space[-1])

    def update_Q(self):
        for idx, (state, action, _) in enumerate(self.memory):
            G = 0 # total return
            if self.pairs_visited[(state, action)] == 0: # since First Visit Monte Carlo
                self.pairs_visited[(state, action)] += 1
                discount = 1
                for t, (_, _, reward) in enumerate(self.memory[idx:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[(state, action)].append(G)

        for state, action, _ in self.memory:
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])

        for state_action in self.pairs_visited.keys():
            self.pairs_visited[state_action] = 0

        self.memory = []

if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    agent = Agent()
    episodes = NO_EPISODES
    ep_hundred = -1 # tracks episodes (splits into 100) - includes episode 0 (hence -1)
    ep_score, ep_no = np.zeros(RECORDINGS), np.zeros(RECORDINGS)

    for i in range(episodes):
        if i % (episodes//RECORDINGS) == 0:
            ep_hundred += 1
            ep_no[ep_hundred] = i
            print("Current episode: {}, current epsilon is: {}".format(i, agent.epsilon))
        observation = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info = env.step(action) # obs (Sum of cards, dealer card, has ace), reward (1.0 or -1.0 if done, else 0), done: terminated?
            agent.memory.append((observation, action, reward))
            observation = observation_
            if reward != 0:
                ep_score[ep_hundred] += reward
        agent.update_Q()

    x = ep_no
    y = ep_score

    plt.plot(x,y)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Average Agent Score')
    plt.show()


    print(agent.Q[(21, 3, True), 0])
    print(agent.Q[(21, 3, True), 1])
    print(agent.Q[(11, 6, False), 0])
    print(agent.Q[(11, 6, False), 1])
    print(agent.Q[(4, 1, False), 0])
    print(agent.Q[(4, 1, False), 1])
