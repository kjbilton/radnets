"""
Tools for performing simple hyperparameter optimization.
"""
import numpy as np
from scipy.stats import loguniform


def generate_log_param(param_range):
    return loguniform(*param_range).rvs()


def generate_param(param_range):
    return np.random.uniform(*param_range)


def make_empty_params():
    return {
        'mean_mda': (np.inf, np.inf), 'lr': 0, 'l2_lambda': 0, 'threshold': 0,
        'neurons': 0}


def make_empty_params_list():
    return {
        'mean_mda': [], 'lr': [], 'l2_lambda': [], 'threshold': [],
        'neurons': []}
