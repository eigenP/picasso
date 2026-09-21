import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_independent_channel_scale_equivariance():
    """
    Testr 🔎: Verify Independent Channel Scale Equivariance of the unmixing algorithm.

    This test validates a critical physical property of the blind source separation:
    Independent Channel Scale Equivariance.

    If we scale input channels independently by a diagonal matrix $S$ (e.g., doubling the gain
    on the green channel while halving the exposure on the red channel), the unmixing matrix
    must transform in a predictable conjugate manner:

    $U_{scaled} = S \\cdot U_{base} \\cdot S^{-1}$

    This guarantees that the underlying unmixing coefficients correctly preserve the
    dimensionless physical ratios of crosstalk, rendering the algorithm insensitive
    to independent channel scaling operations (gain, exposure time, or normalization).
    """
    # 1. Setup: Generate Synthetic Correlated Data
    np.random.seed(42)
    n_pixels = 10_000

    # Independent sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape to (Channels, Y, X) for the API
    mixed = mixed_flat.reshape(2, n_pixels, 1)

    # 2. Compute Baseline Unmixing Matrix
    u_base = compute_unmixing_matrix(
        list(mixed),
        verbose=False,
        max_iters=100, # Increased limit to ensure proper convergence
        quantile=0.0,  # Process all pixels
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Apply Independent Channel Scaling
    # Define a diagonal scaling matrix (e.g., Channel 1 x2.0, Channel 2 x0.5)
    S = np.array([[2.0, 0.0],
                  [0.0, 0.5]])

    # Scale the flattened mixed data and reshape back
    mixed_scaled = (S @ mixed_flat).reshape(2, n_pixels, 1)

    # 4. Compute Scaled Unmixing Matrix
    u_scaled = compute_unmixing_matrix(
        list(mixed_scaled),
        verbose=False,
        max_iters=100, # Consistent increased limit
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Verify the Invariant: U_scaled = S * U_base * S^-1
    S_inv = np.linalg.inv(S)
    expected_u_scaled = S @ u_base @ S_inv

    # Assert equality with relaxed tolerance due to floating-point min/max
    # bounds jitter in dynamic entropy binning calculation.
    np.testing.assert_allclose(
        u_scaled,
        expected_u_scaled,
        atol=1e-2,
        err_msg="Independent Channel Scale Equivariance failed. Scaling channels did not yield the conjugate transformed unmixing matrix."
    )
