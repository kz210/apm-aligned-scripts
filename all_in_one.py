# ============================================================
# ACTIVE PORTFOLIO MANAGEMENT — COMPLETE PYTHON NOTEBOOK
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, solve

np.random.seed(42)

# ============================================================
# 1. FUNDAMENTAL LAW OF ACTIVE MANAGEMENT
# ============================================================

def information_ratio(IC, BR, TC=1.0):
    """
    Computes Information Ratio using the Fundamental Law:
    IR = TC * IC * sqrt(BR)
    IC: INFO Coefficient
    BR: Breadth
    TC: Transfer Coefficient
    """
    return TC * IC * np.sqrt(BR)

# Example
IC = 0.05
BR = 300
print("IR example:", information_ratio(IC, BR))


# ============================================================
# 2. INFORMATION COEFFICIENT (IC) ESTIMATION
# ============================================================

# Synthetic forecasts and realized returns
n_assets = 50
n_periods = 250

forecasts = pd.DataFrame(np.random.normal(0, 0.05, (n_periods, n_assets)))
realized = pd.DataFrame(np.random.normal(0, 0.05, (n_periods, n_assets)))

def estimate_ic(forecasts, realized):
    """
    Cross-sectional IC for each period.
    """
    return forecasts.corrwith(realized, axis=1)

IC_series = estimate_ic(forecasts, realized)
print("Mean IC:", IC_series.mean())

plt.figure(figsize=(10,4))
plt.plot(IC_series)
plt.title("Daily Information Coefficient")
plt.axhline(IC_series.mean(), color='red')
plt.show()


# ============================================================
# 3. BREADTH SIMULATION
# ============================================================

def simulate_breadth(IC=0.05, BR_values=[50, 100, 200, 400]):
    IRs = [information_ratio(IC, BR) for BR in BR_values]
    return pd.DataFrame({"BR": BR_values, "IR": IRs})

df_br = simulate_breadth()
print(df_br)

plt.figure(figsize=(6,4))
plt.plot(df_br["BR"], df_br["IR"], marker='o')
plt.title("IR vs Breadth")
plt.xlabel("Breadth")
plt.ylabel("Information Ratio")
plt.show()


# ============================================================
# 4. FACTOR MODEL CONSTRUCTION
# ============================================================

# Synthetic factor exposures
k_factors = 3
X = np.random.normal(0, 1, (n_assets, k_factors))

# Factor covariance
F = np.array([[0.04, 0.01, 0.00],
              [0.01, 0.03, 0.00],
              [0.00, 0.00, 0.02]])

# Idiosyncratic variances
D = np.random.uniform(0.01, 0.03, n_assets)

def factor_covariance(X, F, D):
    return X @ F @ X.T + np.diag(D)

Sigma = factor_covariance(X, F, D)
print("Covariance matrix shape:", Sigma.shape)


# ============================================================
# 5. ACTIVE RISK CALCULATION
# ============================================================

def active_risk(weights, Sigma):
    return np.sqrt(weights.T @ Sigma @ weights)

# Example random active weights
w = np.random.normal(0, 0.01, n_assets)
print("Active risk:", active_risk(w, Sigma))


# ============================================================
# 6. ACTIVE PORTFOLIO OPTIMISATION
# ============================================================

def optimal_active_weights(alpha, Sigma, lam=1.0):
    """
    Solves: max_w alpha^T w - (lam/2) w^T Sigma w
    Closed form: w* = (1/lam) * Sigma^{-1} alpha
    """
    return solve(lam * Sigma, alpha)

# Synthetic alpha vector
alpha = np.random.normal(0, 0.02, n_assets)

w_opt = optimal_active_weights(alpha, Sigma, lam=1.0)
print("Optimal weights (first 10):", w_opt[:10])

print("Optimal active risk:", active_risk(w_opt, Sigma))
print("Optimal IR:", (alpha @ w_opt) / active_risk(w_opt, Sigma))


# ============================================================
# 7. PERFORMANCE ATTRIBUTION
# ============================================================

# Synthetic factor returns
factor_returns = np.array([0.01, -0.005, 0.002])

def factor_attribution(weights, exposures, factor_returns):
    """
    Contribution = (w^T X) * f
    """
    factor_exposure = weights @ exposures
    return factor_exposure * factor_returns

factor_contrib = factor_attribution(w_opt, X, factor_returns)
print("Factor contributions:", factor_contrib)

# Specific return attribution
specific_returns = np.random.normal(0, 0.01, n_assets)
specific_contrib = w_opt * specific_returns
print("Specific contribution (sum):", specific_contrib.sum())


# ============================================================
# 8. SUMMARY OUTPUT
# ============================================================

print("\n=== SUMMARY ===")
print("Mean IC:", IC_series.mean())
print("Optimal IR:", (alpha @ w_opt) / active_risk(w_opt, Sigma))
print("Factor contribution:", factor_contrib.sum())
print("Specific contribution:", specific_contrib.sum())
