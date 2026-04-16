from scipy.optimize import minimize_scalar

def f(x):
    return (x - 2.5)**2

res = minimize_scalar(f, bounds=(0, 100), method='bounded')
print(res)
