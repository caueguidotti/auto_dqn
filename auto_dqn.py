#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File        : auto_dqn.py
# Project     : Project
# Created By  : Caue Guidotti
# Created Date: 7/11/2021
# =============================================================================
"""
This module is used to apply MIES to optimize hyperparameters of a DQN and the
ANN architecture used by it
"""
# =============================================================================
import os
from MIES import MIES
from MIES import Spaces
from DQN import env_trainer
from functools import partial
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def dqn_evaluation(gym_env, args):
    """
    Evaluation function used by MIES
    :param gym_env: OpenAI Gym evironment name
    :param args: MIES solution candidate
    :return: Mean reward of evaluated episodes
    """
    # unwraps candidate solution
    num_neurons_layer1 = args[0]
    num_neurons_layer2 = args[1]
    num_neurons_layer3 = args[2]
    activation_layer1 = args[3]
    activation_layer2 = args[4]
    activation_layer3 = args[5]
    adam_learning_rate = args[6]
    gamma = args[7]
    
    # Defines dict used by DQN Agent (see dqn_agent's make_model method)
    dqn_model_layers = [
        {'num_neurons': num_neurons_layer1, 'activation': utils.num_to_activation(activation_layer1)},
        {'num_neurons': num_neurons_layer2, 'activation': utils.num_to_activation(activation_layer2)},
        {'num_neurons': num_neurons_layer3, 'activation': utils.num_to_activation(activation_layer3)},
    ]
    
    # defines the trainer - fixed batch size, num of episodes and max step of gym execution. Don't render env to gain time.
    trainer = env_trainer.EnvTrainer(gym_env, dqn_model_layers, batch_size=32, num_episodes=100, render_env=False,
                                     max_steps=250, adam_lr=adam_learning_rate, gamma=gamma, verbose=True)
    trainer.train()  # starts training
    
    # defines the evaluator (10% of episodes used for training)
    evaluator = env_trainer.EnvEval(gym_env, dqn_model_layers, num_episodes=int(trainer.num_episodes/10), render_env=False,
                                    max_steps=250, adam_lr=adam_learning_rate, gamma=gamma,
                                    model_name=trainer.model_name, verbose=True)
    return_val = evaluator.evaluate()  # evaluate method returns the mean value of all episodes
    return return_val


def dqn_evaluation_simple(gym_env, args):
    """
    The Simple evaluation function used by MIES
    :param gym_env: OpenAI Gym evironment name
    :param args: MIES solution candidate
    :return: Mean reward of evaluated episodes
    """
    # similar to dqn_evaluation - but used for simple MIES execution (ie. debug)
    num_neurons_layer1 = args[0]
    num_neurons_layer2 = args[1]
    activation_layer1 = args[2]
    activation_layer2 = args[3]

    dqn_model_layers = [
        {'num_neurons': num_neurons_layer1, 'activation': utils.num_to_activation(activation_layer1)},
        {'num_neurons': num_neurons_layer2, 'activation': utils.num_to_activation(activation_layer2)},
    ]

    trainer = env_trainer.EnvTrainer(gym_env, dqn_model_layers, batch_size=32, num_episodes=50, render_env=False,
                                     max_steps=250, adam_lr=0.001, gamma=0.95, verbose=True)
    trainer.train()

    evaluator = env_trainer.EnvEval(gym_env, dqn_model_layers, num_episodes=5, render_env=False,
                                    max_steps=250, adam_lr=0.001, gamma=0.95,
                                    model_name=trainer.model_name, verbose=True)
    return_val = evaluator.evaluate()
    return return_val


if __name__ == '__main__':

    simple = False

    if not simple:
        # Create search space
        # neurons num = [1, 30]
        space_num_neurons_layer1 = Spaces.IntegerSpace(lower_limit=1, upper_limit=30)
        space_num_neurons_layer2 = Spaces.IntegerSpace(lower_limit=1, upper_limit=30)
        space_num_neurons_layer3 = Spaces.IntegerSpace(lower_limit=1, upper_limit=30)
        
        # activation function = [0, 3]
        space_activation_layer1 = Spaces.IntegerSpace(lower_limit=0, upper_limit=3)
        space_activation_layer2 = Spaces.IntegerSpace(lower_limit=0, upper_limit=3)
        space_activation_layer3 = Spaces.IntegerSpace(lower_limit=0, upper_limit=3)
        
        # learning rate = [10^-5, 10^-3]
        space_adam_learning_rate = Spaces.ContinuousSpace(lower_limit=10**-5, upper_limit=10**-3)
        
        # gamma = [0.8, 0.99]
        space_gamma = Spaces.ContinuousSpace(lower_limit=0.8, upper_limit=0.99)
        
        # combine spaces and form search space
        search_space = (space_num_neurons_layer1*space_num_neurons_layer2*space_num_neurons_layer3*
                        space_activation_layer1*space_activation_layer2*space_activation_layer3*
                        space_adam_learning_rate*space_gamma)
        
        # create the base individual
        individual_base = Spaces.Individual(search_space=search_space)
        
        # set available gyms, chooses mountain car and run MIES
        gym_envs = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0']
        # set evaluation function to mountain car
        eval_fn = partial(dqn_evaluation, gym_envs[2])
        # mies-(4,6) - max 200 epochs - maximize objective function 
        mies = MIES.MIES(individual_base, parents_num=4, offsprings_num=6, evaluation_function=eval_fn, max_epoch=200,
                         minimize=False)
        mies.evolve()
    else:
        # similar to above, but reduced search space (only integer) and less epochs - used for debugging
        space_num_neurons_layer1 = Spaces.IntegerSpace(lower_limit=20, upper_limit=30)
        space_num_neurons_layer2 = Spaces.IntegerSpace(lower_limit=20, upper_limit=30)

        space_activation_layer1 = Spaces.IntegerSpace(lower_limit=0, upper_limit=3)
        space_activation_layer2 = Spaces.IntegerSpace(lower_limit=0, upper_limit=3)

        search_space = (space_num_neurons_layer1 * space_num_neurons_layer2  *
                        space_activation_layer1 * space_activation_layer2)
        individual_base = Spaces.Individual(search_space=search_space)

        eval_fn = partial(dqn_evaluation_simple, 'MountainCar-v0')
        mies = MIES.MIES(individual_base, parents_num=4, offsprings_num=6, evaluation_function=eval_fn, max_epoch=50,
                         minimize=False)
        mies.evolve()
