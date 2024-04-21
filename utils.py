import numpy as np

def uniformly_random_point_on_n_sphere(n):
    p = np.random.normal(size=n+1)
    return p/(np.linalg.norm(p))


