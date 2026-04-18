import numpy as np
from scipy.optimize import minimize_scalar

def func(alpha):
    return (alpha - 0.5)**2

try:
    res = minimize_scalar(func, bounds=(0.0, 1.0), bracket=(0.0, 0.4), method='bounded')
    print("Bounded bracket works?")
except Exception as e:
    print("Bounded bracket failed:", e)
