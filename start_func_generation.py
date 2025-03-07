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

def window_func3_(x):
    if x >= 0.1 and x < 0.2 or x >= 0.8 and x < 0.9 :
        return 1
    else:
        return 0
    
def normal_dist_func(x, mu=0.5, sig=0.1):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

def harmonic_func(x, n=10, end_x=1):
    return np.sin((np.pi * n * x) / end_x)


window_func = np.vectorize(window_func_)
window_func2 = np.vectorize(window_func2_)
window_func3 = np.vectorize(window_func3_)