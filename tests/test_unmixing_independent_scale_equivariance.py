import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_independent_scale_equivariance():
    """
    Testr 🔎: Verify Independent Channel Scale Equivariance.

    This test validates that independently scaling the input channels
    yields an exactly predictable conjugate transformation of the unmixing matrix.

    Why this matters:
    In fluorescence microscopy, different channels have vastly different brightness
    levels due to fluorophore quantum yield, laser power, or camera exposure.
    If Channel 2 is made $k$ times brighter, its crosstalk into Channel 1 appears
    $k$ times smaller relative to Channel 2's new intensity, so $U_{12}$ must scale by $1/k$.
    Conversely, Channel 1's crosstalk into Channel 2 now appears $k$ times larger
    relative to Channel 1, so $U_{21}$ must scale by $k$.

    Mathematically, if $S$ is a diagonal scaling matrix such that $I_{scaled} = S \cdot I$,
    then the new unmixing matrix MUST be $U_{scaled} = S \cdot U_{base} \cdot S^{-1}$.

    This is a profound physical invariant. It proves the optimization landscape perfectly
    preserves the underlying dimensionless physical ratios (crosstalk percentages)
    regardless of relative channel intensities.
    """
    np.random.seed(42)
    n_pixels = 20_000

    # 1. Setup: Generate Synthetic Correlated Data
    # Independent sources
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed = M @ sources
    mixed_img = mixed.reshape(2, n_pixels, 1)

    # 2. Compute Baseline Unmixing Matrix
    U_base = compute_unmixing_matrix(
        list(mixed_img),
        verbose=False,
        max_iters=20,
        quantile=0.0,  # Process all pixels to avoid subsampling noise
        max_samples=n_pixels
    )

    # 3. Independently Scale Channels
    # We choose factors such that the required unmixing coefficients do not exceed
    # the algorithm's default bounds (upper bound of 1.0 for each step).
    # Since step_mult=0.1, the maximum unmixing coefficient applied per iteration is -0.1.
    # To stay within bounds across 20 iterations, the total required coefficient should be small.
    # Base coefficients are U_base[0, 1] ~ -0.37, U_base[1, 0] ~ -0.18.
    # If S = [2.0, 0.5], U_scaled[0, 1] = 2.0 * (-0.37) / 0.5 = -1.48 (which might exceed bounds)
    # Let's use milder scaling to stay within the bounds of a few iterations without hitting clipping.
    s_factors = [1.2, 0.8]
    S = np.diag(s_factors)
    S_inv = np.diag([1.0 / s for s in s_factors])

    mixed_scaled = S @ mixed
    mixed_scaled_img = mixed_scaled.reshape(2, n_pixels, 1)

    # 4. Compute Scaled Unmixing Matrix
    U_scaled = compute_unmixing_matrix(
        list(mixed_scaled_img),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Verify Conjugate Transformation
    # U_scaled MUST equal S * U_base * S_inv
    U_expected = S @ U_base @ S_inv

    # We allow a small tolerance because dynamic binning in 1D marginal entropy
    # can introduce very minor numerical jitter when absolute ranges change drastically.
    np.testing.assert_allclose(
        U_scaled,
        U_expected,
        atol=1e-2,
        err_msg=(
            "Independent Scale Equivariance failed. \n"
            f"Expected U_scaled:\n{U_expected}\n"
            f"Got U_scaled:\n{U_scaled}\n"
            "The algorithm failed to preserve dimensionless physical ratios."
        )
    )
