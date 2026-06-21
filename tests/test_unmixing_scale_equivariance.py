import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_independent_scale_equivariance():
    """
    Testr 🔎: Verify Independent Channel Scale Equivariance.

    This test validates that:
    When individual input channels are scaled independently by a diagonal matrix $S$
    (e.g., simulating vastly different exposure times, camera gains, or fluorophore brightnesses),
    the resulting unmixing matrix undergoes a predictable conjugate transformation:
    $U_{scaled} = S \cdot U_{base} \cdot S^{-1}$.

    Why this matters:
    In real fluorescence microscopy, channels almost never have the same absolute
    intensity scale. The Red channel might be 5x brighter than the Green channel
    simply due to laser power or quantum yield. The unmixing algorithm must extract
    the true dimensionless bleed-through ratios regardless of these absolute scales.

    By verifying that $U_{scaled}$ maps exactly to $S \cdot U_{base} \cdot S^{-1}$,
    we prove that the algorithm successfully normalizes intensity differences and
    operates purely on the underlying mutual information overlap.
    """
    # 1. Setup: Generate Synthetic Correlated Data
    np.random.seed(42)
    n_pixels = 50_000

    # Independent sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])

    # 2. Compute Baseline Unmixing Matrix
    mixed_base_flat = M @ sources
    mixed_base = mixed_base_flat.reshape(2, n_pixels, 1)

    u_base = compute_unmixing_matrix(
        list(mixed_base),
        verbose=False,
        max_iters=20,
        quantile=0.0, # Use all pixels to avoid stochastic selection noise from shifting percentiles
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Independently Scaled Data
    # S scales Channel 1 by 2.0 and Channel 2 by 5.0
    S = np.diag([2.0, 5.0])

    # Scale the already mixed data
    # Equivalent to: Channel 1 * 2.0, Channel 2 * 5.0
    mixed_scaled_flat = S @ mixed_base_flat
    mixed_scaled = mixed_scaled_flat.reshape(2, n_pixels, 1)

    # 4. Compute Unmixing Matrix on Scaled Data
    u_scaled = compute_unmixing_matrix(
        list(mixed_scaled),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Independent Scale Equivariance
    # The expected result is the conjugate transformation of the base matrix
    # expected = S @ U_base @ S^-1
    S_inv = np.linalg.inv(S)
    expected_u_scaled = S @ u_base @ S_inv

    # We use a relaxed tolerance (1e-2) because empirical min/max bounds in 1D
    # histogram bin width calculations can introduce minor numeric jitter when values scale dynamically.
    np.testing.assert_allclose(
        u_scaled,
        expected_u_scaled,
        atol=1e-2,
        err_msg="Independent Scale Equivariance failed. Scaling channels independently did not result in the expected conjugate transformation of the unmixing matrix."
    )
