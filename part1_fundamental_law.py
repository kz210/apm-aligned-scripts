import numpy as np

def information_ratio(IC, BR, TC=1.0):
    return TC * IC * np.sqrt(BR)
