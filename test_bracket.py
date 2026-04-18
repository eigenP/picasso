import numpy as np
from scipy.optimize import minimize_scalar

def func(alpha):
    return (alpha - 0.5)**2

# Bracket with (0.0, 0.4, 1.0)
res = minimize_scalar(func, bounds=(0.0, 1.0), bracket=(0.0, 0.4, 1.0), method='bounded')
print(res)
