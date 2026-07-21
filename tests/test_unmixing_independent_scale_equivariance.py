import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_independent_scale_equivariance():
    """
    Testr 🔎: Verify Independent Channel Scale Equivariance of the unmixing algorithm.

    This test validates that independently scaling input channels by a diagonal matrix $S$
    results in a predictable conjugate transformation of the unmixing matrix:
    $U_{scaled} = S \\cdot U_{base} \\cdot S^{-1}$.

    In fluorescence microscopy, this means that if one channel is imaged with a longer
    exposure time or higher gain than another, the fundamental unmixing relationship
    is preserved (up to the scaling factors), and the unmixed result will still yield
    independent channels with the original biological dimensionless ratios.

    This property ensures the algorithm correctly normalizes pairwise relationships
    and is invariant to arbitrary independent scalar multipliers.
    """
    # 1. Setup: Generate Synthetic Correlated Data
    np.random.seed(42)
    n_pixels = 50_000

    # Independent sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels) + 0.1
    s2 = np.random.gamma(2, 2, n_pixels) + 0.1
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape to (Channels, Pixels, 1) for the API
    mixed = mixed_flat.reshape(2, n_pixels, 1)

    # 2. Compute Baseline Unmixing Matrix
    u_base = compute_unmixing_matrix(
        list(mixed),
        verbose=False,
        max_iters=50,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Independently Scaled Data
    # Scale channel 1 by 2.5x and channel 2 by 0.5x
    scales = np.array([2.5, 0.5])
    S = np.diag(scales)
    S_inv = np.diag(1.0 / scales)

    mixed_scaled_flat = S @ mixed_flat
    mixed_scaled = mixed_scaled_flat.reshape(2, n_pixels, 1)

    # 4. Compute Unmixing Matrix on Scaled Data
    u_scaled = compute_unmixing_matrix(
        list(mixed_scaled),
        verbose=False,
        max_iters=50,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Independent Channel Scale Equivariance
    expected_u_scaled = S @ u_base @ S_inv

    np.testing.assert_allclose(
        u_scaled,
        expected_u_scaled,
        atol=2e-2,
        err_msg="Independent Channel Scale Equivariance failed. Scaling channels independently did not result in the expected conjugate transformation."
    )
