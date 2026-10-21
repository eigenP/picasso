import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_independent_channel_scale_equivariance():
    """
    Testr 🔎: Verify Independent Channel Scale Equivariance.

    This test validates that independently scaling input channels by a diagonal matrix S
    results in a predictable conjugate transformation of the unmixing matrix:
    $U_{scaled} = S \cdot U_{base} \cdot S^{-1}$.

    Why this matters:
    In physical microscopy, channels often have different integration times, detector
    gains, or fluorophore brightness. The algorithm must be equivariant to these independent
    scaling factors. It should find the exact same relative physical unmixing ratios
    (adjusted for the new scale), meaning the underlying physical information is perfectly
    preserved regardless of channel-specific brightness.
    """
    np.random.seed(42)
    n_pixels = 20_000

    # 1. Generate Synthetic Correlated Data
    # Independent sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([
        [1.0, 0.4],
        [0.2, 1.0]
    ])
    mixed_flat = M @ sources

    # Reshape to (Channels, Pixels, 1) for the API
    mixed = mixed_flat.reshape(2, n_pixels, 1)

    # 2. Compute Baseline Unmixing Matrix
    u_base = compute_unmixing_matrix(
        list(mixed),
        verbose=False,
        max_iters=100, # Increased iteration limit for proper convergence
        quantile=0.0,
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity)."

    # 3. Create Independently Scaled Data
    # Scale Ch 1 by 2.0, Ch 2 by 5.0
    s_factors = np.array([2.0, 5.0])
    S = np.diag(s_factors)

    # Scale the channels
    mixed_scaled_flat = S @ mixed_flat
    mixed_scaled = mixed_scaled_flat.reshape(2, n_pixels, 1)

    # 4. Compute Unmixing Matrix on Scaled Data
    u_scaled = compute_unmixing_matrix(
        list(mixed_scaled),
        verbose=False,
        max_iters=100, # Increased iteration limit for proper convergence
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Equivariance
    # The expected result is the conjugate transformation: S * U_base * S_inv
    S_inv = np.linalg.inv(S)
    expected_u_scaled = S @ u_base @ S_inv

    # Use a relaxed tolerance (atol=0.05) due to floating point min/max bounds jitter
    # in 1D marginal entropy histogram calculation across different scales.
    # The statistical bias threshold also depends on bins / (2*N), which can vary slightly.
    np.testing.assert_allclose(
        u_scaled,
        expected_u_scaled,
        atol=0.05,
        err_msg="Independent Channel Scale Equivariance failed. Scaling channels changed physical ratios incorrectly."
    )
