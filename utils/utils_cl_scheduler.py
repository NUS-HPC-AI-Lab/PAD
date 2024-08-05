import numpy as np

def linear_cl_scheduler_acse(t, a, b):
    return min(1, 1.0 * a + (1 - a) * 1.0 * t / b)

def root_cl_scheduler(t, a, b):
    return min(1, np.sqrt(a**2 + (1 - a**2) * t / b))

def geometric_cl_scheduler(t, a, b):
    return min(1, 2**(np.log2(a) - np.log2(a * t / b)))

def linear_cl_scheduler_desc(t, t_s, r, max_r):
    if t < t_s:
        return 0.0
    else:
        return min(max_r, (t - t_s) * r)


