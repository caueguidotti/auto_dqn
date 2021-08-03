#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File        : collect_results.py
# Project     : Project
# Created By  : Caue Guidotti
# Created Date: 7/17/2021
# =============================================================================
"""
This module is used to process results obtained with auto_dqn module
"""
import os
from natsort import natsorted  # natsort is a third party library (available on PyPI)
from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt

# auto-generated results folder
results_folder = 'results'

# MIES results within auto-generated results folder
results_base_name = '2021_07_08__00_04_12_274400__MIES'

results_files_all = natsorted(glob(os.path.join(results_folder, results_base_name + '_alls*')))
results_files_pop = natsorted(glob(os.path.join(results_folder, results_base_name + '_cur_pop_*')))

with open(results_files_all[0], "r") as fp:
    all_individuals = list(map(lambda i: {'score': i[0], 'params': i[1]}, json.load(fp)))
all_individuals = sorted(all_individuals, key=lambda item: item['score'])

# reads all individuals from each population file
pops_individuals = []
for result_file in results_files_pop:
    with open(result_file, "r") as fp:
        individuals = sorted(map(lambda i: {'score': i[0], 'params': i[1]}, json.load(fp)),
                             key=lambda item: item['score'])
        pops_individuals.append(individuals)

# find best individuals
best_individual_each_epoch = [epoch_individuals[-1] for epoch_individuals in pops_individuals]
best_individual = max(best_individual_each_epoch, key=lambda ind: ind['score'])

best_each_pop = []
parameters_choices = np.array([])
for pop_individuals in pops_individuals:
    best_epoch_score = pop_individuals[-1]['score']
    best_each_pop.append(best_epoch_score)

    individuals_params = np.array([individual['params'] for individual in pop_individuals]).T
    if parameters_choices.shape[0] == 0:
        parameters_choices = individuals_params[np.newaxis, :]
    else:
        parameters_choices = np.vstack([parameters_choices, individuals_params[np.newaxis, :]])

# plot results
# convergence
plt.plot(best_each_pop, 'b')
plt.xlabel('Epoch')
plt.ylabel('Mean Reward')
plt.savefig('mies_best_individual.eps')
plt.savefig('mies_best_individual.png')
plt.show()

# plot parameters evolution
params_to_plot_list = [[0, 1, 2], [3, 4, 5], [6, 7]]
params_names = [['Neurons Layer 1', 'Neurons Layer 2', 'Neurons Layer 3'],
                ['Activation Layer 1', 'Activation Layer 2', 'Activation Layer 3'],
                ['Learning Rate', 'Discount Factor']]
# styles
ind_colors = ['k', 'c', 'm', 'y']
param_line_style = ['-', '--', ':']
for params_to_plot, names in zip(params_to_plot_list, params_names):
    params = parameters_choices[:, params_to_plot, :]

    for param_idx, ind_param in enumerate(params.T.swapaxes(0, 1)):
        #plt.figure()
        for ind_idx, ind in enumerate(ind_param):
            plt.plot(ind, color=ind_colors[ind_idx], linestyle=param_line_style[param_idx])
        #plt.show()

# plot only best individual paramters evolution
param_names = ['Neurons Layer 1', 'Neurons Layer 2', 'Neurons Layer 3',
               'Activation Layer 1', 'Activation Layer 2', 'Activation Layer 3',
               'Learning Rate', 'Discount Factor']
best_ind_params = parameters_choices.T[-1, :, :]
for ind_param, name in zip(best_ind_params, param_names):
    plt.figure()
    plt.plot(ind_param)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.savefig(f'mies_best_param_{name}.eps')
    plt.savefig(f'mies_best_param_{name}.png')
    plt.show()
