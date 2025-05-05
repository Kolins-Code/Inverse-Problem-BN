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

def sinh_func(x):
    return np.sinh(np.pi) * np.sin(np.pi * x)

def abs_func(x, end_x = 1):
    return -np.abs(x - end_x / 2) + end_x / 2

def normal_sources(x):
    #if x >= 0 and x < 1/3:
    #    return 0.5 * normal_dist_func(x + 0.4, sig=0.04)
    #elif x >= 1/3 and x < 2/3:
    #    return 0.9 * normal_dist_func(x, sig=0.04)
    #else:
    #    return 0.7 * normal_dist_func(x - 0.4, sig=0.04)
    return 0.5 * normal_dist_func(x + 0.3, sig=0.04) + 0.9 * normal_dist_func(x, sig=0.04) + 0.7 * normal_dist_func(x - 0.3, sig=0.04)
    

window_func = np.vectorize(window_func_)
window_func2 = np.vectorize(window_func2_)
window_func3 = np.vectorize(window_func3_)
#normal_sources = np.vectorize(normal_sources_)