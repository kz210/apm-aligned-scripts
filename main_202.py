import numpy as np
import pandas as pd

import edhec_risk_kit_202 as erk



"""Performing Sharpe Style Analysis
The key to obtaining the weights is our old friend, the quadriatic optimizer. We are asking the optimizer to find 
the weights that minimizes the square of the difference between the observed series and the returns of a benchmark 
portfolio that holds the explanatory building blocks in those same weights. 
This is equivalent to minimizing the tracking error between the two return series."""
ind = erk.get_ind_returns()["2000":]
mgr_r = 0.3*ind["Beer"] + .5*ind["Smoke"] + 0.2*np.random.normal(scale=0.15/(12**.5), size=ind.shape[0])
weights = erk.style_analysis(mgr_r, ind)*100
weights.sort_values(ascending=False).head(6).plot.bar()

coeffs = erk.regress(mgr_r, ind).params*100
coeffs.sort_values().head()
coeffs.sort_values(ascending=False).head(6).plot.bar()


"""
Style Drift: Time Varying Exposures using Style Anaylsis
One of the most common ways in which Sharpe Style Analysis can be used is to measure style drift. 
If you run the style analysis function over a rolling window of 1 to 5 years, you can extract changes in the style exposures of a manager.
"""
brka_m = pd.read_csv("brka_m.csv", index_col=0, parse_dates=True).to_period('M')
mgr_r_b = brka_m["2000":]["BRKA"]
weights_b = erk.style_analysis(mgr_r_b, ind)
weights_b.sort_values(ascending=False).head(6).round(4)*100

brk2009 = brka_m["2009":]["BRKA"]
ind2009 = ind["2009":]
erk.style_analysis(brk2009, ind2009).sort_values(ascending=False).head(6).round(4)*100
