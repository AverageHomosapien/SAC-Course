import pybullet_envs
import gym
import os
import numpy as np
from DDPG import ddpg_run
from DQN import dqn_run
from TD3 import td3_run
from PPO import ppo_run
from SAC import sac_run

number_of_continuous_runs = 1
continuous_envs = ['LunarLanderContinuous-v2', 'MountainCarContinuous-v0', 'InvertedPendulumBulletEnv-v0'] #Hopper-v2
continuous_env_steps = [20000, 20000, 20000]
continuous_runs = [ddpg_run, td3_run, sac_run] #ppo_run,
continuous_action_spaces = [2, 1, 1]
continuous_obs_spaces = [(8,), (2,), (5,)]

number_of_discrete_runs = 1
discrete_envs = ['LunarLander-v2', 'MountainCar-v0']
discrete_env_steps = [20000, 20000]
discrete_runs = [dqn_run]

if __name__ == '__main__':
    env_steps_continuous = list(zip(continuous_action_spaces, continuous_obs_spaces, continuous_envs, continuous_env_steps))
    for times in range(number_of_continuous_runs):
        for idx, run in enumerate(continuous_runs):
            run(actions=env_steps_continuous[idx][0], obs=env_steps_continuous[idx][1],
                env_id=env_steps_continuous[idx][2], total_games=env_steps_continuous[idx][3], run=idx)

    for times in number_of_discrete_runs:
        for idx, run in enumerate(discrete_runs):
            run(env_id=env_steps_continuous[idx][0], total_games=env_steps_continuous[idx][1])
