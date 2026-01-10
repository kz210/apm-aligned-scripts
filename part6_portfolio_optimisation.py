import numpy as np


def optimal_active_weights(alpha, Sigma, lam=1.0):
    return np.linalg.solve(lam * Sigma, alpha)
