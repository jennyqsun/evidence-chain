import numpy as np

def sigmoid(x, alpha=1, beta=0):
    return 1 / (1 + np.exp(-(alpha*x - beta)))


def get_abs_evidence(x):
    return np.abs(x.cumsum(axis=1))