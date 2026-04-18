import numpy as np
from picasso.unmixing import mutual_information
from fast_histogram import histogram2d, histogram1d

np.random.seed(42)
N = 10000
bins = 100
x = np.random.gamma(2, 2, N)
y = x * 0.5 + np.random.gamma(2, 2, N)

c_xy = histogram2d(x, y, bins, [(x.min(), x.max()), (y.min(), y.max())])
c_x = histogram1d(x, bins, (x.min(), x.max()))
c_y = histogram1d(y, bins, (y.min(), y.max()))

k_xy = np.sum(c_xy > 0)
k_x = np.sum(c_x > 0)
k_y = np.sum(c_y > 0)

print(f"K_XY={k_xy}, K_X={k_x}, K_Y={k_y}")
df = k_xy - k_x - k_y + 1
print(f"Effective df={df}")
