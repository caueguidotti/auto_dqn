#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File        : benchmark_functions.py
# Project     : Project
# Created By  : Caue Guidotti
# Created Date: 7/7/2021
# =============================================================================
"""
Provides benchmark functions for debugging MIES
"""
# =============================================================================
import numpy as np


def styblinski_tang_function(*args):
    args = np.asarray(args)
    # Styblinski–Tang function
    return np.sum(args ** 4 - 16 * args ** 2 + 5 * args) / 2


def sphere_function(*args):
    args = np.asarray(args)
    return np.sum(args ** 2)
