import numpy as np

def window_func_(x):
    if x > 0.4 and x < 0.6:
        return 1
    else:
        return 0
    
def window_func2_(x):
    if x >= 0 and x < 0.6:
        return 1
    else:
        return 0
    
def normal_dist_func(x, mu=0.5, sig=0.1):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

window_func = np.vectorize(window_func_)
window_func2 = np.vectorize(window_func2_)
