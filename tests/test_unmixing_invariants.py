import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_invariants():
    """
    Testr 🔎: Verify fundamental physical invariants of the unmixing algorithm.

    This test validates that the algorithm respects:
    1. **Scale Invariance**: Multiplying the input intensities by a constant factor $k$
       must yield the exact same unmixing matrix. The unmixing coefficients are
       dimensionless ratios of intensities, and should not depend on absolute units
       (e.g., exposure time or gain).

       Mathematically: $U(k \cdot I) = U(I)$

    2. **Permutation Equivariance**: Swapping the order of input channels (e.g., Red/Green)
       must result in a corresponding permutation of the unmixing matrix. Specifically,
       for a 2-channel system, swapping inputs results in the transpose of the unmixing
       matrix (due to the diagonal=1 constraint).

       Mathematically: If $I_{perm} = P \cdot I$, then $U(I_{perm}) = U(I)^T$ for $P = [[0, 1], [1, 0]]$.

    These properties are "hard" invariants—they should hold exactly (within floating point precision)
    regardless of the data distribution, provided the data is valid.
    """
    # 1. Setup: Generate Synthetic Correlated Data
    # We need data with real correlations so the unmixing matrix is non-trivial (not Identity).
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
        mixed,
        verbose=False,
        max_iters=20, # Sufficient for convergence on this simple case
        quantile=0.0, # Use all pixels to avoid stochastic selection noise
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Verify Scale Invariance
    # Scale by a factor that isn't a power of 2 to avoid trivial float exponents
    scale_factor = 3.14159

    u_scaled = compute_unmixing_matrix(
        mixed * scale_factor,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Assert exact equality (or very high precision)
    # The algorithm uses histograms which are invariant if bin edges scale proportionally.
    np.testing.assert_allclose(
        u_scaled,
        u_base,
        atol=1e-10, # Strict tolerance
        err_msg=f"Scale Invariance failed. Scaling by {scale_factor} changed the unmixing matrix."
    )

    # 4. Verify Permutation Equivariance
    # Swap channels 0 and 1
    mixed_perm = mixed[::-1, :, :] # (2, N, 1) -> (2, N, 1) with channels swapped

    u_perm = compute_unmixing_matrix(
        mixed_perm,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # The expected result is the Transpose of the base matrix
    # Why? See derivation in Testr journal/thought process.
    # U_base = [[1, -a], [-b, 1]] unmixes (Ch0, Ch1)
    # U_perm unmixes (Ch1, Ch0).
    # It finds coefficient to remove Ch0 from Ch1 -> -b (stored at [0, 1])
    # It finds coefficient to remove Ch1 from Ch0 -> -a (stored at [1, 0])
    # So U_perm = [[1, -b], [-a, 1]] which is U_base.T
    expected_u_perm = u_base.T

    np.testing.assert_allclose(
        u_perm,
        expected_u_perm,
        atol=1e-10,
        err_msg="Permutation Equivariance failed. Swapping channels did not transpose the unmixing matrix."
    )
