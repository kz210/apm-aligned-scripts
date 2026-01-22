import os
import pandas as pd
import edhec_risk_kit_201 as erk

brka_d = pd.read_csv("data/brka_d_ret.csv", parse_dates=True, index_col=0)
print(brka_d.head())

brka_m = brka_d.resample('M').apply(erk.compound).to_period('M')
print(brka_m.head())
#brka_m.to_csv("brka_m.csv")
fff = erk.get_fff_returns()
print(fff.head())


import statsmodels.api as sm
import numpy as np
brka_excess = brka_m["1990":"2012-05"] - fff.loc["1990":"2012-05", ['RF']].values #r - rf
mkt_excess = fff.loc["1990":"2012-05",['Mkt-RF']] #rm - rf
exp_var = mkt_excess.copy()
exp_var["Constant"] = 1
lm = sm.OLS(brka_excess, exp_var).fit()
print(lm.summary())


"""
The CAPM benchmark interpretation
This implies that the CAPM benchmark consists of 46 cents in T-Bills and 54 cents in the market. 
i.e. each dollar in the Berkshire Hathaway portfolio is equivalent to 46 cents in T-Bills and 54 cents in the market.
Relative to this, the Berkshire Hathaway is adding (i.e. has 𝛼 of) 0.61% (per month!) although the degree of statistica significance is not very high.
Now, let's add in some additional explanatory variables, namely Value and Size
"""
exp_var["Value"] = fff.loc["1990":"2012-05", ['HML']]
exp_var["Size"] = fff.loc["1990":"2012-05", ['SMB']]
exp_var.head()

lm = sm.OLS(brka_excess, exp_var).fit()
print(lm.summary())

