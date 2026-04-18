import numpy as np
from scipy.optimize import minimize_scalar
from picasso.unmixing import mutual_information, minimize_mi

np.random.seed(42)
N = 10000
x = np.random.gamma(2, 2, N)
y = x * 0.5 + np.random.gamma(2, 2, N)

def func(alpha):
    return mutual_information(x, y - alpha * x, bins=100)

print("minimize_scalar (bounded) with default settings:")
res_brent = minimize_scalar(func, bounds=(0.0, 1.0), method='bounded')
print(res_brent)

print("\nminimize_mi from code:")
res_my_brent = minimize_mi(x, y, init_alpha=0.4, bins=100)
print(res_my_brent)
