import numpy as np
import gym

class Agent():
    def __init__(self, gamma=0.99):
        self.V = {} # estimate of state value
        self.sum_space = [i for i in range(4, 22)] # bust above 21 (not really part of sum space)
        self.dealer_show_card_space = [i+1 for i in range(10)]
        self.ace_space = [False, True]

        self.action_space = [0, 1] # stick or hit
        self.state_space = []

        self.returns = {}
        self.states_visited = {} # First visit Monte Carlo
        self.memory = []
        self.gamma = gamma

        self.init_vals()

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total, card, ace)] = 0
                    self.returns[(total, card, ace)] = []
                    self.states_visited[(total, card, ace)] = 0
                    self.state_space.append((total, card, ace))
        # self.V # Dictionary - can use tuple as key {(4, 1, False): 0, (4, 1, True): 0, ..}
        # self.returns # Dictionary of tuple, list for returns {(4, 1, False): [], (4, 1, True): [], ..}
        # self.states_visited # Dictonary noting which states have been visited (0,1) {(4, 1, False): 0, (4, 1, True): 0, ..}
        # self.state_space # List denoting the different states [(4, 1, False), (4, 1, True), ..]

        # APPEND ALL RETURNS TO self.returns
        # AVERAGE THESE RETURNS IN self.V (could have it so only the 'touched' states update averages)

    def policy(self, state):
        total, _, _ = state
        action = 0 if total >= 20 else 1
        return action

    def update_V(self):
        for idx, (state, _) in enumerate(self.memory):
            G = 0 # total return
            if self.states_visited[state] == 0: # since Firs Visit Monte Carlo
                self.states_visited[state] += 1
                discount = 1
                for t, (_, reward) in enumerate(self.memory[idx:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        for state in self.state_space:
            self.states_visited[state] = 0

        self.memory = []

if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    agent = Agent()
    episodes = 50000
    for i in range(episodes):
        if i % (episodes//100) == 0:
            print("starting episode: " + str(i))
        observation = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info = env.step(action) # obs (Sum of cards, dealer card, has ace), reward (1.0 or -1.0 if done, else 0), done: terminated?
            agent.memory.append((observation, reward))
            observation = observation_
        agent.update_V()
    print(agent.V[(21, 3, True)])
    print(agent.V[(4, 1, False)])
