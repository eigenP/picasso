import numpy as np
from picasso.unmixing import compute_unmixing_matrix, mutual_information

def calculate_total_pairwise_mi(data):
    n_channels = data.shape[0]
    total_mi = 0.0
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            mi = mutual_information(data[i], data[j], bins=100)
            total_mi += mi
    return total_mi

np.random.seed(123)
n_pixels = 50_000

s1 = np.random.gamma(2, 2, n_pixels)
s2 = np.random.gamma(2, 2, n_pixels)
s3 = np.random.gamma(2, 2, n_pixels)
sources = np.stack([s1, s2, s3])

# Highly correlated 3x3 mixing matrix
M = np.array([
    [1.0, 0.8, 0.6],
    [0.7, 1.0, 0.5],
    [0.6, 0.4, 1.0]
])
mixed_flat = M @ sources
mixed_input = mixed_flat.reshape(3, n_pixels, 1)

mats = compute_unmixing_matrix(
    mixed_input,
    verbose=False,
    max_iters=50,
    quantile=0.0,
    max_samples=n_pixels,
    return_iters=True,
    step_mult=0.5 # larger step size to induce oscillation
)

initial_mi = calculate_total_pairwise_mi(mixed_flat)
print(f"Initial MI: {initial_mi:.6f}")
prev_mi = initial_mi

increases = 0
for t, mat in enumerate(mats):
    unmixed_flat = mat @ mixed_flat
    current_mi = calculate_total_pairwise_mi(unmixed_flat)
    if current_mi > prev_mi:
        print(f"Iter {t+1:2d} MI: {current_mi:.6f} INCREASED from {prev_mi:.6f}")
        increases += 1
    else:
        print(f"Iter {t+1:2d} MI: {current_mi:.6f} decreased")
    prev_mi = current_mi

print(f"Total increases: {increases}")
