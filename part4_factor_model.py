def factor_covariance(X, F, D):
    return X @ F @ X.T + np.diag(D)
