import os
import pybullet_envs
import gym
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.functional as F
from torch.distributions import Normal
from utils import ReplayBuffer, plot_learning_curve

"""
I noticed something interesting with SAC in the Lunar Lander environment -
I let it play 2000 games with a batch size of 64 and reward scd.cale of 2 (first try,
I have not tuned either of these). It got up to an average score around 120 at
the 900 game mark then slowly leaked down to 115 average by the 2000 mark.

I was disappointed until I looked at the actual renderings. The algorithm learns
to land quite well, then randomly fires the thrusters until the game is terminated
at 1000 steps. If you save a trained model, switch to deterministic and turn learning off,
SAC performs on par with TD3 (trained similarly and re-run with no action noise or learning).

The perceived benefit from keeping entropy high (resulting in thrusters randomly
firing even when mu is near zero) must be greater than the negative reward.

If there were some reward very late in the game, SAC would have found it.

"""

# V, Q, Policy networks
# V takes states -> outputs scalar value
# Critic takes states & actions, outputs scalar value (incorp actions first layer and use concat)
# Actor takes states -> outputs mean and sigma
    # Constrain sigma with clamp (0,1) or sigmoid
# Pytorch normal distribution for actions
# Reparameterisation trick (sample vs rsample) - boolean for Reparameterisation?

# Log prob has dimensions n_actions, need to sum/mean
# Multivariate normal distribution doesn't seem to work
# Can also copy over the functionality to save networks
# Check appendix for hyperparameters

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True, deterministic=False):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        if not deterministic:
            return action, log_probs
        else:
            return mu, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, env_id, alpha=0.0003, beta=0.0003, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, deterministic=False):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False, deterministic=deterministic)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
#InvertedPendulumBulletEnv
# seperate method for running the network so that it can be called from run_agents
#def sac_run(actions=None, obs=None, env_id='HopperBulletEnv-v0', total_runs=1000000, run=0):
def sac_run(actions=None, obs=None, env_id='MountainCarContinuous-v0', total_runs=50000, run=6): #4 (0.8 reward scale), 6 is updated alpha and betas
#def sac_run(actions=None, obs=None, env_id='LunarLanderContinuous-v2', total_runs=500000, run=0):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    runs = total_runs
    total_actions = env.action_space.shape[0] if actions == None else actions
    obs_space = env.observation_space.shape if obs == None else obs

    agent = Agent(alpha=0.0007, beta=0.0007, reward_scale=3, env_id=env_id,
                input_dims=obs_space, tau=0.01, max_size=50000, gamma=0.999,
                env=env, batch_size=512, layer1_size=64, layer2_size=64,
                n_actions=total_actions)
    file = 'plots/sac_' + env_id + "_"+ str(total_runs) + '_run_' + str(run) + '_games'
    file2 = 'plots/sac_eval_' + env_id + "_"+ str(total_runs) + '_run_' + str(run) + '_games'

    best_score = env.reward_range[0]
    scores = []
    steps = []
    eval_scores = []
    eval_steps = []
    total_steps = 0 # tracking steps
    eval_step = 0 # tracking evaluation steps
    total_eval_steps = 150000 # total evaluation steps

    while True:
        step = 0
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            score += reward
            step += 1
            total_steps += 1

            if total_steps % 10000 == 0:
                print("current steps are {}, last score was {}".format(total_steps, scores[-1]))

        scores.append(score)
        steps.append(step)
        if total_steps >= runs:
            agent.save_models()
            print('run {}, steps {}, score {}, env {}'.format(total_steps, step, score, env_id))
            zipped_list = list(zip(scores, steps))
            df = pd.DataFrame(zipped_list, columns=['Scores', 'Steps'])
            df.to_csv(file + '.csv')
            break
        elif total_steps % 100 == 0: # Chance of saving on 200 steps
            agent.save_models()
            print('run {}, steps {}, score {}, env {}'.format(total_steps, step, score, env_id))
            zipped_list = list(zip(scores, steps))
            df = pd.DataFrame(zipped_list, columns=['Scores', 'Steps'])
            df.to_csv(file + '.csv')

    while True:
        eval_score = 0
        step = 0
        eval_observation = eval_env.reset()
        eval_done = False
        while not eval_done:
            eval_action = agent.choose_action(eval_observation)
            eval_observation_, eval_reward, eval_done, eval_info = eval_env.step(eval_action, deterministic=True)
            eval_score += eval_reward
            eval_observation = eval_observation_
            eval_step += 1
            step += 1
            if eval_step % 10000 == 0:
                print("current steps are {}".format(total_steps))

        eval_scores.append(eval_score)
        eval_steps.append(step)

        if eval_step >= total_eval_steps:
            print('eval steps {}, score {}, env {}'.format(eval_step, eval_score, env_id))
            zipped_list2 = list(zip(eval_scores, eval_steps))
            df2 = pd.DataFrame(zipped_list2, columns=['Scores', 'Steps'])
            df2.to_csv(file2 + '.csv')
            break
        elif eval_step % 100 == 0: # Chance of saving on 500 steps
            print('eval steps {}, score {}, env {}'.format(eval_step, eval_score, env_id))
            zipped_list2 = list(zip(eval_scores, eval_steps))
            df2 = pd.DataFrame(zipped_list2, columns=['Scores', 'Steps'])
            df2.to_csv(file2 + '.csv')

# environments with large negative rewards don't work (e.g. LunarLander)
if __name__ == '__main__':
    sac_run()
