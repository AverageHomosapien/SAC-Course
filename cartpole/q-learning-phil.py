import numpy as np
import gym
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, lr, gamma, n_actions, state_space, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.n_actions = n_actions

        self.state_space = state_space
        self.actions = [i for i in range(self.n_actions)]

        self.Q = {}
        self.init_Q()

    def init_Q(self):
        for state in self.state_space:
            for action in self.action_space:
                self.Q[(states, self.actions)] = 0.0

    def max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.action_space])
        action = np.argmax(actions)
        return action

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.max_action(state)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_end else self.eps_end

    def learn(self, state, action, reward, state_):
        a_max = self.max_action(state_)

        self.Q[(state, action)] = self.Q[(state, action)] + self.lr * (reward + \
                                self.gamma * self.Q[(state_, a_max)] -
                                self.Q[(state, action)])

class CartPoleStateDigitizer():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)
        self.bins = n_bins
        self.states = self.get_state_space()

    def get_state_space(self):
        states = []
        for i in range(self.bins+1): # works if bins are all same size
            for j in range(self.bins+1):
                for k in range(self.bins+1):
                    for l in range(self.bins+1):
                        states.append((i, j, k, l))
        return states

    def digitize(self, observation):
        x, x_dot, theta, theta_dot = observation
        cart_x = int(np.digitize(x, self.position_space))
        cart_xdot = int(np.digitize(x_dot, self.velocity_space))
        pole_theta = int(np.digitize(theta, self.pole_angle_space))
        pole_thetadot = int(np.digitize(theta_dot, self.pole_velocity_space))

        return (cart_x,, cart_xdot, pole_theta, pole_thetadot)

def plot_learning_curve(scores, x):
    

if __name__ == "__main__":
