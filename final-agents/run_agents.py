import pybullet_envs
import gym
import os
import numpy as np
from DDPG import ddpg_run
from DQN import dqn_run
from TD3 import td3_run
from PPO import ppo_run
from SAC import sac_run

continuous_envs = ['LunarLanderContinuous-v2', 'MountainCarContinuous-v0',
                    'InvertedPendulumBulletEnv-v0', 'Hopper-v1']
continuous_env_steps = [50000, 50000,
                        50000, 50000]
continuous_runs = [ddpg_run, td3_run, ppo_run, sac_run]
continous_action_spaces = [4, 2, 1, ]
continuous_obs_spaces = [8, 4, 5, ]

discrete_envs = ['LunarLander-v2', 'MountainCar-v0']
discrete_env_steps = [40000, 40000]
discrete_runs = [dqn_run]

if __name__ == '__main__':
    env_steps_continuous = list(zip(continuous_envs, continuous_env_steps))
    for idx, run in enumerate(continuous_runs):
        run(env_id=env_steps_continuous[idx][0], total_runs=env_steps_continuous[idx][1])

    for idx, run in enumerate(discrete_runs):
        run(env_id=env_steps_continuous[idx][0], total_runs=env_steps_continuous[idx][1])
