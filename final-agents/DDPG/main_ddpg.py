import gym
import numpy as np
import pandas as pd
from ddpg_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    #def ddpg_run(actions=None, obs=None, env_id='LunarLanderContinuous-v2', test_model=False, total_games=20000, run=1):
    #def ddpg_run(actions=None, obs=None, env_id='MountainCarContinuous-v0', test_model=False, total_games=20000, run=0):
    #def ddpg_run(actions=None, obs=None, env_id='HopperBulletEnv-v0', test_model=False, total_games=100000, run=1):

    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=256, fc1_dims=256, fc2_dims=256,
                    n_actions=env.action_space.shape[0])
    n_games = 20000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        #if avg_score > best_score:
        #    best_score = avg_score
			
        if i % 20 == 0:
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    df = pd.DataFrame(score_history)
    df.to_csv(filename + '.csv')



"""

    env = gym.make(env_id)
    n_games = total_games
    load_checkpoint = test_model
    total_actions = env.action_space.shape[0] if actions == None else actions
    obs_space = env.observation_space.shape if obs == None else obs

    agent = DDPGAgent(alpha=0.0001, beta=0.001, input_dims=obs_space,
                tau=0.001, batch_size=256, fc1_dims=256, fc2_dims=256,
                n_actions=total_actions, env_id=env_id)

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.save_models()

    for i in range(n_games):
        steps = 0
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            steps += 1
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        #if avg_score > best_score:
        #    best_score = avg_score
        if i % 20 == 0:
            if not load_checkpoint:
                agent.save_models()

        print('episode {} score {} trailing 100 games avg {} steps {} env {}'.format(
        i, score, avg_score, steps, env_id))

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        file = 'plots/ddpg_' + env_id + "_"+ str(n_games) + '_run_' + str(run) + '_games'
        filename = file + '.png'
        plot_learning_curve(x, score_history, filename)

if __name__ == "__main__":
    ddpg_run()
"""
