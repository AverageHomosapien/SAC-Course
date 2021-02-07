import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class OUActionNoise(ActionNoise):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # noise = OUActionNoise()
        # current_noise = noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 if not None else np.zeros_like(self.mu)


class ReplayBuffer():
    def __init__(self, min_sample=200, max_size=10e6, minibatch_size=32, initial_state = None):
        self.queue = deque() if initial_state == None else deque([initial_state])
        self.max_size = max_size
        self.minibatch_size = minibatch_size

    def store_transition(self, observation):
        if self.is_full():
            self.queue.popleft()
        self.queue.append(observation)

    def is_full(self):
        if len(self.queue) >= max_size:
            return True
        return False

    # the only reason I can see memory attributes stored seperately is for the calculation
    # .. since if doing specific calculation in batch, and tuples stored together,
    # .. all tuples will need unpacked
    def sample(self, minibatch_size):
        if len(self.queue) < min_sample:
            return []
        return np.random.choice(self.queue, size=self.minibatch_size)


def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
