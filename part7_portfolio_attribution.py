def factor_attribution(weights, exposures, factor_returns):
    return (weights @ exposures) * factor_returns
