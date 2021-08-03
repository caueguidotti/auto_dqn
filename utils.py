#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File        : utils.py
# Project     : Project
# Created By  : Caue Guidotti
# Created Date: 7/11/2021
# =============================================================================
"""
This is an auxiliary script
"""
# =============================================================================


def num_to_activation(num):
    """
        Converts num to respective activation function
        :param num: activation function num
        :return: activation function string
    """
    d = {
        0: 'LeakyReLU',
        1: 'relu',
        2: 'tanh',
        3: 'sigmoid',
    }

    return d[num]
