#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File        : MIES.py
# Project     : Project
# Created By  : Caue Guidotti
# Created Date: 7/6/2021
# =============================================================================
"""
Implements the Mixed-integer evolution strategies optimizer class
"""
# =============================================================================
import os
import numpy as np
import copy
import heapq
import multiprocessing
import json
from datetime import datetime


class MIES(object):
    """
    MIES Class implementation

    TODO: Allows for an external termination function
    """
    def __init__(self, individual_base, parents_num, offsprings_num, evaluation_function,
                 plus_selection=False, max_epoch=10, minimize=True):
        """
        MIES Class constructor
        :param individual_base: A base individual (see Spaces.Individual)
        :param parents_num: Mu value (number of parents)
        :param offsprings_num: Lambda value (number of offspring)
        :param evaluation_function: Objective function - optimization target
        :param plus_selection: (boolean) uses plus selection if true, comma selection if false
        :param max_epoch: Number of epochs that will be evaluated
        :param minimize: True for minimize the objective function, False for maximize
        """
        self.individual_base = individual_base
        self.parents_num = parents_num
        self.offsprings_num = offsprings_num
        self.evaluation_function = evaluation_function
        self.plus_selection = plus_selection
        self.max_epoch = max_epoch
        self.minimize = minimize

        # TODO: make these arguments
        self.evaluate_in_parallel = True  # if TRUE uses multiprocessing package to call multiple evaluation
        self.results_dir = 'results'  # dir that results will be stored
        self.results_file = datetime.now().strftime(f"%Y_%m_%d__%H_%M_%S_%f__MIES")  # result files suffix
        if not os.path.isdir(self.results_dir):
            # creates results dir if doesnt exist
            os.mkdir(self.results_dir)

        self.epoch = 1
        self.parents = None
        self.offsprings = [None]*self.offsprings_num
        self.initialized = False
        self.ancestries_evaluation = []
        self.epoch_evaluation = []
        self.epoch_best = []
        self.unique_individuals = 0

    def initialize(self):
        """
        Creates first parents population and calls individuals initialization method
        """
        print('Initializing population')
        self.parents = [copy.deepcopy(self.individual_base) for _ in range(self.parents_num)]
        for parent in self.parents:
            parent.initialize()

    def evolve(self):
        """
        Implements the evolution strategy algorithm

        0 - t <- 0
        1 - Initialize population P(t) E I
        2 - Evaluate mu initial individual_spaces with objective function f
        3 - while Termination criteria not fulfilled
        4 - for all i E {1, ..., lambda} do
        5 - Choose uniform randomly parents c_i1 and c_i2 from P(t) (rept ok)
        6 - x_i <- mutate(recombine(c_i1, c_i2))
        7 - Q(t) <- Q(t) U {x_i} (expands Q)
        8 - end for
        9 - P(t+1) <- mu individual_spaces with best objective function value from P(t) U Q(t) (plus) or Q(t) (comma)
        10 - t <- t+1
        11 - end while
        :return:
        """

        if not self.initialized:
            self.initialize()
            self.evaluate(self.parents)
            self.add_to_ancestry_heap(self.parents)
            self.epoch_best.append(self.ancestries_evaluation[0].objective_value)

        while not self.has_reach_termination():
            self.epoch_evaluation = []

            for offspring_idx in range(self.offsprings_num):
                p1, p2 = np.random.randint(0, self.parents_num), np.random.randint(0, self.parents_num)
                offspring = copy.deepcopy(self.parents[p1])
                offspring.recombine(self.parents[p2])
                offspring.mutate()
                self.offsprings[offspring_idx] = offspring

            print('Offsprings Generated')
            self.evaluate(self.offsprings)
            self.select()

            self.epoch += 1
            self.save()

    def evaluate(self, individuals):
        """
        Call evaluation function for desired individuals parameters
        :param individuals: Individuals to be evaluated
        """

        # obtain individuals candidate solution (ie. characteristics)
        individual_characteristics = list(map(lambda individual: getattr(individual, 'characteristics'),
                                              individuals))
        print(f'\tEvaluating population - {"multi" if self.evaluate_in_parallel else "single"}-thread')
        if self.evaluate_in_parallel:
            proc_pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
            objective_values = proc_pool.map(self.evaluation_function, individual_characteristics)
        else:
            objective_values = list(map(self.evaluation_function, individual_characteristics))

        for ind_idx, (individual, objective_value) in enumerate(zip(individuals, objective_values)):
            print(f'\t\tIndividual #{ind_idx+1} | Objective Value: {objective_value}')
            individual.set_objective_value(objective_value)

    def select(self):
        """
        Selects next generation parents
        """
        population = self.offsprings + self.parents if self.plus_selection else self.offsprings
        for individual in population:
            heapq.heappush(self.epoch_evaluation, individual)
        self.add_to_ancestry_heap(population)

        self.parents = heapq.nsmallest(self.parents_num, self.epoch_evaluation) if self.minimize \
            else heapq.nlargest(self.parents_num, self.epoch_evaluation)
        self.epoch_best.append(self.parents[0].objective_value)

    def add_to_ancestry_heap(self, individuals):
        """
        Add new individuals to list of all individuals (saves progress)
        :param individuals: Individuals to be added
        """
        for individual in individuals:
            if individual not in self.ancestries_evaluation:  # evaluates if individual is already in list
                heapq.heappush(self.ancestries_evaluation, individual)
                self.unique_individuals += 1

    def has_reach_termination(self):
        """
        Implements the termination function
        """
        return self.epoch >= self.max_epoch

    def save(self):
        """
        Saves current progress.
        List of all individuals evaluated is saved as well as each epoch individuals
        """
        with open(os.path.join(self.results_dir, self.results_file + '_alls.json'), "w") as fp:
            info_dump = map(lambda ind: (ind.objective_value, ind.characteristics.tolist()), self.ancestries_evaluation)
            json.dump(list(info_dump), fp, indent=4)
        with open(os.path.join(self.results_dir, self.results_file + f'_cur_pop_{self.epoch}.json'), "w") as fp:
            info_dump = map(lambda ind: (ind.objective_value, ind.characteristics.tolist()), self.parents)
            json.dump(list(info_dump), fp, indent=4)


if __name__ == '__main__':
    # Following is used for debugging
    from benchmark_functions import sphere_function, styblinski_tang_function
    from Spaces import SearchSpace, Individual

    space = SearchSpace(boundaries=[(-1.0, 1.0), (-3, -1), (-5, 5)])
    individual_base = Individual(search_space=space)

    mies = MIES(individual_base, parents_num=4, offsprings_num=10, evaluation_function=styblinski_tang_function, max_epoch=100)
    mies.evolve()

    exit(0)
