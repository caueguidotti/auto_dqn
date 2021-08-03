#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File        : Spaces.py
# Project     : Project
# Created By  : Caue Guidotti
# Created Date: 7/6/2021
# =============================================================================
"""
This module provides the basic classes to create an individual and
its search space.
Possible space types are: Continuous or Integer

TODO: implement nominal (discrete) space
"""
# =============================================================================

import numpy as np

POSSIBLE_SPACE_TYPES = ['i', 'c']


class Individual(object):
    """
    Individual class
    An individual may be recombined with another to change its characteristics and its search space state
    An individual may also go through mutation

    TODO: Perhaps receive the evaluation function in this class
    """
    def __init__(self, search_space):
        """
        An individual is created
        :param search_space: Individual's search space
        """
        assert isinstance(search_space, SearchSpace), 'Individual must be supplied with a object of type SearchSpace'
        self.search_space = search_space
        self.characteristics = None
        self.objective_value = None
        self.tau_r_global = None
        self.tau_r_local = None
        self.tau_i_global = None
        self.tau_i_local = None

    def __lt__(self, other_individual):
        """
        Overloads < operator for Individual class. Used for heapq.
        """
        return self.objective_value < other_individual.objective_value

    def __eq__(self, other_individual):
        """
        Overloads == operator. Used, for instance, to check if an individual is already present in a population.
        """
        return (self.characteristics == other_individual.characteristics).all()

    def initialize(self):
        """
        Initialize the individual
        """

        # obtains a random candidate solution
        self.characteristics = self.get_sample()

        # Defines learning rates
        # TODO - enable single and multi step-size mode
        # TODO - allows learning rate to be set externally
        if 'c' in self.search_space.get_types():
            len_c = len(self.search_space.idx_dict['c'])
            self.tau_r_global = 1.0 / np.sqrt(2.0 * len_c)
            self.tau_r_local = 1.0 / np.sqrt(2 * np.sqrt(len_c)) if len_c > 1 else 0

        if 'i' in self.search_space.get_types():
            len_i = len(self.search_space.idx_dict['i'])
            self.tau_i_global = 1.0 / np.sqrt(2.0 * len_i)
            self.tau_i_local = 1.0 / np.sqrt(2 * np.sqrt(len_i)) if len_i > 1 else 0

    def get_sample(self) -> np.ndarray:
        """
        Obtains a random candidate solution
        """
        sample = []
        for space in self.search_space.spaces:
            sample += [space.get_sample()]
        return np.asarray(sample, dtype='object')

    def get_continuous_characteristics(self):
        """
        Get continuous candidate solutions
        """
        return self.characteristics[self.search_space.idx_dict['c']]

    def set_continuous_characteristics(self, values):
        """
        Set individual continuous candidate solution
        :param values: New continuous parameters values
        """
        assert len(self.search_space.idx_dict['c']) == len(values), 'Characteristics size must match values size'
        self.characteristics[self.search_space.idx_dict['c']] = values

    def get_integer_characteristics(self):
        """
        Get integer candidate solutions
        """
        return self.characteristics[self.search_space.idx_dict['i']]

    def set_integer_characteristics(self, values):
        """
        Set individual integer candidate solution
        :param values: New integer parameters values
        """
        assert len(self.search_space.idx_dict['i']) == len(values), 'Characteristics size must match values size'
        self.characteristics[self.search_space.idx_dict['i']] = values

    def set_objective_value(self, value):
        """
        Set the corresponding evaluation value for this individual characteristics
        :param value: Evaluated value
        :return:
        """
        self.objective_value = value

    def recombine(self, other_individual,
                  characteristics_global_recombination=True,
                  strategy_pars_global_recombination=True,
                  strategy_par_random_weight=False):
        """
        This individual can be recombined with another. This individual will be changed.

        In MIES, characteristics associated to the solution uses dominant (or discrete) recombination.
            Intermediate recombination is used for strategy parameters
        :param other_individual: another individual
        :param characteristics_global_recombination: (bool) use global recombination or not for solution params
        :param strategy_pars_global_recombination: (bool) use global recombination or not for strategy params
        :param strategy_par_random_weight: Use random weight instead of mean (0.5)
        :return:
        """
        # Discrete recombination for characteristics
        if characteristics_global_recombination:
            chosen_characteristics = np.nonzero(np.random.rand(len(self.characteristics)) > 0.5)
            self.characteristics[chosen_characteristics] = other_individual.characteristics[chosen_characteristics]
        else:
            chosen_characteristics = np.random.rand() > 0.5
            if chosen_characteristics:
                self.characteristics = other_individual.characteristics

        # Intermediate recombination for strategy parameters
        strategy_par_weight = 0.5 if not strategy_par_random_weight else np.random.rand()
        for space, other_space in zip(self.search_space.spaces, other_individual.search_space.spaces):
            if strategy_par_random_weight and strategy_pars_global_recombination:
                strategy_par_weight = np.random.rand()

            combined_strategy_par = space.get_strategy_parameter() * strategy_par_weight + \
                                    other_space.get_strategy_parameter() * (1 - strategy_par_weight)
            space.set_strategy_parameter(combined_strategy_par)

    def mutate(self):
        """
        Applies the mutation algorithm
        """
        continuous_spaces = self.search_space.get_continuous_spaces()
        integer_spaces = self.search_space.get_integer_spaces()

        if continuous_spaces is not None:
            # mutate continuous spaces characteristics
            spaces_sigma = continuous_spaces.get_strategy_parameters()
            spaces_sigma *= np.exp(self.tau_r_global * np.random.normal(0, 1) +
                                   self.tau_r_local * np.random.normal(0, 1, len(spaces_sigma)))

            continuous_characteristics = self.get_continuous_characteristics()
            continuous_characteristics += spaces_sigma * np.random.normal(0, 1, len(continuous_characteristics))

            continuous_spaces.set_strategy_parameters(spaces_sigma)
            self.set_continuous_characteristics(continuous_characteristics)

        if integer_spaces is not None:
            # mutate integer spaces characteristics
            spaces_eta = integer_spaces.get_strategy_parameters()
            spaces_eta *= np.exp(self.tau_i_global * np.random.normal(0, 1) +
                                 self.tau_i_local * np.random.normal(0, 1, len(spaces_eta)))
            spaces_eta = spaces_eta.clip(min=1)

            integer_characteristics = self.get_integer_characteristics()

            psi = 1 - ((spaces_eta / len(integer_characteristics)) /
                       (1 + np.sqrt(1 + (spaces_eta / len(integer_characteristics))**2)))
            (u1, u2) = (np.random.rand(), np.random.rand())
            g1 = np.floor(np.log(1 - u1) / np.log(1. - psi)).astype(int)
            g2 = np.floor(np.log(1 - u2) / np.log(1. - psi)).astype(int)
            integer_characteristics += g1 - g2

            integer_spaces.set_strategy_parameters(spaces_eta)
            self.set_integer_characteristics(integer_characteristics)

        self.boundary_handling()

    def boundary_handling(self):
        """
        Keep solution within boundaries
        """
        # TODO improve boundary handling - clipping adds bias?
        lower_bounds, upper_bounds = list(zip(*self.search_space.get_spaces_boundaries()))
        self.characteristics = np.clip(self.characteristics, lower_bounds, upper_bounds)


class SearchSpace(object):
    """
    Defines a search space which is comprised of multiple spaces
    """
    def __init__(self, spaces=None, boundaries=None, strategy_pars=None):
        """
        A search space may be formed by providing Spaces objects directly OR boundaries and strategy pars, as space
        objects already contains its boundaries.
        If boundaries and strategy pars are provided, space objects are created.

        :param spaces: Spaces in this search space
        :param boundaries: Spaces boundaries
        :param strategy_pars: Spaces strategy parameters
        """

        self._spaces = spaces
        self._boundaries = boundaries
        self._strategy_pars = strategy_pars
        self._parameters_consolidation()  # reads parameters

        # following is a dict of possible space types to respective indexes on self.spaces list
        self.idx_dict = {space_type: [] for space_type in POSSIBLE_SPACE_TYPES}
        self.spaces = []

        if self._spaces is None:
            for space_idx, ((lower_bound, upper_bound), strategy_par) in enumerate(zip(self._boundaries,
                                                                                       self._strategy_pars)):
                # Make sure types are consistent. Lower limit must be lower than upper limit. After identifying type,
                # specific space class is instantiated, and index added to dict of index reference
                assert type(lower_bound) == type(upper_bound), f'Types of boundaries differs: {lower_bound} ({type(lower_bound)}) != {upper_bound} ({type(upper_bound)}) '
                assert lower_bound < upper_bound, f'Lower limit ({lower_bound}) must be lower than upper limit ({upper_bound})'

                if type(lower_bound) == int:
                    self.spaces.append(IntegerSpace(lower_bound, upper_bound, strategy_par))
                elif type(lower_bound) == float:
                    self.spaces.append(ContinuousSpace(lower_bound, upper_bound, strategy_par))
                self.idx_dict[self.spaces[-1].type].append(space_idx)
        else:
            self.spaces = self._spaces
            for space_idx, space in enumerate(self.spaces):
                self.idx_dict[space.type].append(space_idx)

        self.spaces = np.asarray(self.spaces, dtype='object')

    def _parameters_consolidation(self):
        """
        Treats the case of providing either spaces or boundaries.
        If strategy pars are not provided, default are used.
        """
        spaces = self._spaces
        boundaries = self._boundaries
        strategy_pars = self._strategy_pars

        if spaces is None:
            assert boundaries is not None, 'Should either supply a list of spaces or space boundaries to SearchSpace'

            assert hasattr(boundaries, '__iter__') and not isinstance(boundaries, str), \
                'boundaries parameter must be iterable'

            # force strategy_pars to be iterable
            if not hasattr(strategy_pars, '__iter__') and not isinstance(strategy_pars, str):
                strategy_pars = [strategy_pars]

            # making sure we have a list of tuples on boundaries
            if hasattr(boundaries[0], '__iter__') and not isinstance(boundaries[0], str):
                boundaries = [tuple(boundary) for boundary in boundaries]
            else:
                # probably single list or tuple
                boundaries = [tuple(boundaries)]

            # making sure we have tuples with two elements
            for boundary in boundaries:
                assert len(boundary) % 2 == 0, 'args should represent boundaries (pairs):' \
                                               '(lower_limit1, upper_limit1, lower_limit2, upper_limit2)'

            # making sure that, if strategy pars was supplied, it matches the number of spaces boundaries
            if strategy_pars[0] is not None:
                assert len(boundaries) == len(strategy_pars), 'Length of boundaries must match strategy pars'
            else:
                strategy_pars = strategy_pars*len(boundaries)
        else:
            # force strategy_pars to be iterable
            if not hasattr(spaces, '__iter__') and not isinstance(spaces, str):
                spaces = [spaces]

        self._spaces = spaces
        self._boundaries = boundaries
        self._strategy_pars = strategy_pars

    def __mul__(self, other_search_space):
        """
        Calculates the product of search space objects
        :return:
        """
        if isinstance(other_search_space, SearchSpace):
            return SearchSpace(spaces=self.spaces.tolist() + other_search_space.spaces.tolist())
        elif isinstance(other_search_space, GenericSpace):
            return SearchSpace(spaces=self.spaces.tolist() + [other_search_space])
        else:
            raise TypeError(f"can't multiply SearchSpace with type {type(other_search_space).__name__}")

    def get_spaces_boundaries(self, as_list_of_tuples=True):
        """
        Obtain all spaces boundaries within this search space
        :param as_list_of_tuples: Return as list of tuple if true, single line is False
        """
        boundaries = []
        for space in self.spaces:
            if as_list_of_tuples:
                boundaries.append(space.get_boundary())
            else:
                boundaries.extend(space.get_boundary())
        return boundaries

    def get_strategy_parameters(self):
        """
        Get all search spaces strategy parameters
        :return: strategy parameters list
        """
        strategy_pars = []
        for space in self.spaces:
            strategy_pars += [space.get_strategy_parameter()]
        return np.asarray(strategy_pars)

    def set_strategy_parameters(self, values):
        """
        Set strategy parameters for all search spaces.
        :param values: Values to be set
        """
        assert len(values) == len(self.spaces), 'Number of values must be the same as number of spaces.'
        for space, strategy_par in zip(self.spaces, values):
            space.set_strategy_parameter(strategy_par)

    def get_types(self):
        """
        Return all types of spaces in this search space.
        Possible candidates are defined on ´POSSIBLE_SPACE_TYPES´
        """
        return [space_type for space_type, idxs in self.idx_dict.items() if idxs]

    def get_continuous_spaces(self):
        """
        Return list of continuous spaces
        """
        return self._get_space_type(space_type='c')

    def get_integer_spaces(self):
        """
        Return list of integer spaces
        """
        return self._get_space_type(space_type='i')

    def _get_space_type(self, space_type):
        """
        Get all spaces of a specific type. Possible types are defined on ´POSSIBLE_SPACE_TYPES´
        :param space_type: (str) selected type
        :return: list of spaces with this type
        """
        return SearchSpace(spaces=self.spaces[self.idx_dict[space_type]]) if self.idx_dict[space_type] else None


class GenericSpace(object):
    """
    Implements a interface class of a Space
    """
    def __init__(self, lower_limit, upper_limit):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.type = None

    def get_strategy_parameter(self):
        raise NotImplementedError('GenericSpace should not be used directly. Use either Integer or Continuous space.')

    def set_strategy_parameter(self, value):
        raise NotImplementedError('GenericSpace should not be used directly. Use either Integer or Continuous space.')

    def get_boundary(self):
        return self.lower_limit, self.upper_limit

    def get_sample(self):
        raise NotImplementedError('GenericSpace should not be used directly. Use either Integer or Continuous space.')

    def __mul__(self, other_space):
        if isinstance(other_space, GenericSpace):
            return SearchSpace(spaces=[self, other_space])
        elif isinstance(other_space, SearchSpace):
            return SearchSpace(spaces=[self] + other_space.spaces.tolist())
        else:
            raise TypeError(f"can't multiply GenericSpace with type {type(other_space).__name__}")


class IntegerSpace(GenericSpace):
    """
    Implements the integer space
    """
    def __init__(self, lower_limit, upper_limit, mean_step_size=None):
        super().__init__(int(lower_limit), int(upper_limit))
        self.type = 'i'

        if mean_step_size is not None:
            self.mean_step_size = mean_step_size
        else:
            # default strategy parameter
            self.mean_step_size = 0.05 * (self.upper_limit - self.lower_limit)

    def get_strategy_parameter(self):
        """
        Return this space strategy parameter
        """
        return self.mean_step_size

    def set_strategy_parameter(self, value):
        """
        Set a new value for this space strategy parameter
        :param value: Value to be set
        """
        self.mean_step_size = value

    def get_sample(self):
        """
        Get a random value within this space using a uniform distribution
        :return: Random sample
        """
        return np.random.randint(self.lower_limit, self.upper_limit + 1)


class ContinuousSpace(GenericSpace):
    """
    Implements the continuous space
    """
    def __init__(self, lower_limit, upper_limit, std_dev=None):
        super().__init__(float(lower_limit), float(upper_limit))
        self.type = 'c'

        if std_dev is not None:
            self.std_dev = std_dev
        else:
            # default strategy parameter
            self.std_dev = 0.05 * (self.upper_limit-self.lower_limit)

    def get_strategy_parameter(self):
        """
        Return this space strategy parameter
        """
        return self.std_dev

    def set_strategy_parameter(self, value):
        """
        Set a new value for this space strategy parameter
        :param value: Value to be set
        """
        self.std_dev = value

    def get_sample(self):
        """
        Get a random value within this space using a uniform distribution
        :return: Random sample
        """
        return (self.upper_limit - self.lower_limit) * np.random.rand() + self.lower_limit


if __name__ == '__main__':
    # Following is used for debugging.

    C = ContinuousSpace(-10, 10)
    I = IntegerSpace(0, 10)

    search_space1 = C
    search_space2 = I

    search_space1 *= C
    search_space1 = C * search_space1
    search_space1 = search_space1 * C

    search_space2 *= I
    search_space2 = I * search_space2
    search_space2 = search_space2 * I

    search_space = I * search_space1 * search_space2 * C

    individual1 = Individual(search_space=search_space)
    individual1.initialize()

    individual2 = Individual(search_space=search_space)
    individual2.initialize()

    individual1.recombine(individual2)
    individual1.mutate()
    exit(0)
