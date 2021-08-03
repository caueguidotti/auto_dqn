#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File        : env_trainer.py
# Project     : CT-213
# Created By  : Caue Guidotti
# Created Date: 7/8/2021
# =============================================================================
"""
This module was built for implementing the training and evaluation classes of a
DQNAgent.
"""
# =============================================================================
import os
import gym
from datetime import datetime
import numpy as np
from DQN.dqn_agent import DQNAgent
import tensorflow as tf
import json

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class EnvTrainer:
    """
    EnvTrainer class is used for training the DQNAgent
    """
    def __init__(self, gym_env, dqn_model_layers, batch_size=32, num_episodes=300, render_env=False, max_steps=500,
                 adam_lr=10**-3, gamma=0.98, model_name=None, epsilon=0.5, epsilon_min=0.01, verbose=False):
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.render_env = render_env
        self.max_steps = max_steps
        self.verbose = verbose

        # Initialize gym environment
        self.env = gym.make(gym_env)
        self.state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.env_reward = EnvRewarding(gym_env)

        # Initialize dqn agent
        self.model_name = datetime.now().strftime(f"%Y_%m_%d__%H_%M_%S_%f__{gym_env}") if model_name is None else model_name
        self.agent = DQNAgent(self.state_size, action_size, dqn_model_layers, gamma=gamma, learning_rate=adam_lr,
                              epsilon=epsilon, epsilon_min=epsilon_min, verbose=self.verbose)

        # model data files
        self.progress_folder = 'saved_models'
        if not os.path.isdir(self.progress_folder):
            os.mkdir(self.progress_folder)
        self.model_weights_file = f'{self.model_name}__weights.h5'
        self.model_data_file = f'{self.model_name}__model.h5'

        # training history
        self.return_history = []

    def train(self):
        for episodes in range(1, self.num_episodes + 1):
            # Reset the environment
            state = self.env.reset()
            # This reshape is needed to keep compatibility with Keras
            state = np.reshape(state, [1, self.state_size])
            # Cumulative reward is the return since the beginning of the episode
            cumulative_reward = 0.0
            for time in range(1, self.max_steps):
                if self.render_env:
                    self.env.render()  # Render the environment for visualization

                # Select action
                action = self.agent.act(state)
                # Take action, observe reward and new state
                next_state, reward, done, _ = self.env.step(action)
                # Reshaping to keep compatibility with Keras
                next_state = np.reshape(next_state, [1, self.state_size])
                # allowing modification to reward
                reward = self.env_reward.get_reward(state[0], action, reward, next_state[0], done)

                # Appending this experience to the experience replay buffer
                self.agent.append_experience(state, action, reward, next_state, done)
                state = next_state
                # Accumulate reward
                cumulative_reward = self.agent.gamma * cumulative_reward + reward
                if done:
                    if self.verbose:
                        print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                              .format(episodes, self.num_episodes, time, cumulative_reward, self.agent.epsilon))
                    break

                # We only update the policy if we already have enough experience in memory
                if len(self.agent.replay_buffer) > 2 * self.batch_size:
                    loss = self.agent.replay(self.batch_size)
            self.return_history.append(cumulative_reward)
            self.agent.update_epsilon()

            # Every 20 episodes, save model
            if episodes % 20 == 0:
                self.agent.save(self.progress_folder, self.model_data_file, self.model_weights_file)
                with open(os.path.join(self.progress_folder, self.model_name + '_hist.json'), "w") as fp:
                    json.dump(self.return_history, fp)

        # save @ end
        self.agent.save(self.progress_folder, self.model_data_file, self.model_weights_file)
        with open(os.path.join(self.progress_folder, self.model_name + '_hist.json'), "w") as fp:
            json.dump(self.return_history, fp)


class EnvEval(EnvTrainer):
    """
    EnvEval is used for evaluating the DQNAgent trained network
    """
    def __init__(self, gym_env, dqn_model_layers, batch_size=32, num_episodes=300, render_env=False, max_steps=500,
                 adam_lr=10 ** -3, gamma=0.98, model_name=None, verbose=False):

        assert model_name is not None, 'model name must be supplied in order to load model'
        super().__init__(gym_env, dqn_model_layers, batch_size, num_episodes, render_env, max_steps, adam_lr, gamma,
                         model_name, 0.0, 0.0, verbose=verbose)
        self.agent.load(self.progress_folder, self.model_weights_file)

    def train(self):
        raise Exception('Incorrect class usage. Use EnvTrainer instead')

    def evaluate(self):
        for episodes in range(1, self.num_episodes + 1):
            # Reset the environment
            state = self.env.reset()
            # This reshape is needed to keep compatibility with Keras
            state = np.reshape(state, [1, self.state_size])
            # Cumulative reward is the return since the beginning of the episode
            cumulative_reward = 0.0
            for time in range(1, 500):
                # Render the environment for visualization
                if self.render_env:
                    self.env.render()  # Render the environment for visualization
                # Select action
                action = self.agent.act(state)
                # Take action, observe reward and new state
                next_state, reward, done, _ = self.env.step(action)
                # Reshaping to keep compatibility with Keras
                next_state = np.reshape(next_state, [1, self.state_size])
                # allowing modification to reward
                reward = self.env_reward.get_reward(state[0], action, reward, next_state[0], done)

                state = next_state
                # Accumulate reward
                cumulative_reward = self.agent.gamma * cumulative_reward + reward
                if done:
                    if self.verbose:
                        print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                              .format(episodes, self.num_episodes, time, cumulative_reward, self.agent.epsilon))
                    break
            self.return_history.append(cumulative_reward)

        # save hist @ end
        with open(os.path.join(self.progress_folder, self.model_name + '_eval_hist.json'), "w") as fp:
            json.dump(self.return_history, fp)

        # return the mean value of cummulative reward obtained
        return np.mean(self.return_history)


class EnvRewarding:
    """
    Class used to override the reward automatically generated in a OpenAI Gym Environment
    """
    START_POSITION_CAR = -0.5

    def __init__(self, env_name):
        self.env_name = env_name

    def get_reward(self, state, action, reward, next_state, done):
        if self.env_name == 'MountainCar-v0':
            (position, velocity) = state
            (next_position, _) = next_state

            # positive reward if moved from start position and not null velocity
            reward += (position - EnvRewarding.START_POSITION_CAR) ** 2 + velocity ** 2
            # reward if accelerating to corresponding direction
            reward += 1 if velocity > 0 and action == 2 else 1 if velocity < 0 and action == 0 else 0
            # positive reward if climbed hill
            reward += next_position * -2 if next_position <= -1 else 0
            reward += 50 if next_position >= 0.5 else 0

        return reward


if __name__ == '__main__':
    # Following is used for debugging

    # Comment this line to enable training using your GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

    model_layers_dicts = [
        {'num_neurons': 24, 'activation': 'relu'},
        {'num_neurons': 24, 'activation': 'relu'},
    ]

    env_name = 'MountainCar-v0'

    print('training...')
    env_trainer = EnvTrainer(env_name, model_layers_dicts, batch_size=32, num_episodes=50, render_env=False,
                             max_steps=500, adam_lr=0.001, gamma=0.95, verbose=True)
    env_trainer.train()
    print('return:')
    print(env_trainer.return_history)

    print('evaluating...')
    evaluator = EnvEval(env_name, model_layers_dicts, num_episodes=20, render_env=True,
                        max_steps=500, adam_lr=0.001, gamma=0.95, model_name=env_trainer.model_name, verbose=True)
    return_val = evaluator.evaluate()
    print(f'return: {return_val}')
    print(f'return history: {evaluator.return_history}')
