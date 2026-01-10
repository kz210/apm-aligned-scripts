import pandas as pd

def estimate_ic(forecasts, realized):
    return forecasts.corrwith(realized)
