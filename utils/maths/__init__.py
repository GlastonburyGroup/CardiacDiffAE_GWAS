import numpy as np

def safe_sqrt(x):
    return np.sqrt(np.clip(x, a_min=0, a_max=None))

def safe_exp(x, a_max=88.72284):
    return np.exp(np.clip(x, a_min=None, a_max=a_max))
    #88.72284 is the limit for float32 [np.log(np.finfo(np.float32).max)]
    #709.782712893384 is the limit for float64 [np.log(np.finfo(np.float64).max)]

def safe_exp_float32(x):
    return safe_exp(x, a_max=88.72284)

def safe_exp_float64(x):
    return safe_exp(x, a_max=709.782712893384)

def safe_log(x, a_min=11.1920929e-07):
    return np.log(np.clip(x, a_min=a_min, a_max=None))
    #1.1920929e-07 limit for float32 [np.finfo(np.float32).eps]
    #2.220446049250313e-16 limit for float64 [np.finfo(np.float64).eps]

def safe_log_float32(x):
    return safe_log(x, a_min=1.1920929e-07)

def safe_log_float64(x):
    return safe_log(x, a_min=2.220446049250313e-16)
