import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python_plot import plot_learning_curve

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
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

class Agent():
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
                gamma=0.99):
        self.gamma = gamma
        # Only saving the following for saving f_name for plots
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims)
        self.log_prob = None # don't need a list since only tracking a single value
        ## Since doing TD method instead of Monte Carlo, we don't need a list

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward((state))
        # can also use probabilities = self.actor_critc.forward((state[0])) - above is cleaner
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    # don't need to save state when learning online like this
    # Can be passed into learn function - since in loop
    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.Tensor([state]).to(self.actor_critic.device)
        reward = T.Tensor([reward]).to(self.actor_critic.device)
        state_ = T.Tensor([state_]).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        # int(done) is 1 if true, 0 if false (means that it's 0 for terminal state)
        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value

        actor_loss = -self.log_prob * delta
        critic_loss = delta ** 2
        # SUM LOSSES AND BACKPROPEGATE
        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()


EPISODES = 2000

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = Agent(5e-6, env.observation_space.shape[0], env.action_space.n, 2000, 1500)
    fname = 'actor-critic' + 'lunar_lander_lr' + str(agent.lr) + '_' \
            + str(EPISODES) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(EPISODES):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)
