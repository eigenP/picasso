import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_bounds():
    """
    Testr 🔎: Verify the Physical Bounds Constraint (Positivity) of the unmixing algorithm.

    This test validates that the algorithm enforces strict physical reality over pure
    mathematical optimization.

    In fluorescence microscopy, crosstalk (mixing) between channels is strictly additive:
    photons from dye A bleed into channel B. Therefore, the unmixing process must subtract
    signals, requiring the off-diagonal coefficients of the unmixing matrix to be non-positive
    (<= 0).

    If presented with heavily anti-correlated signals (where one channel is high when the
    other is low), a purely statistical algorithm minimizing Mutual Information might try to
    "unmix" them by adding them together (yielding positive off-diagonal coefficients).
    However, this implies "negative photon crosstalk", which is physically impossible.

    This test feeds the algorithm an explicitly adversarial, anti-correlated input.
    We assert that the algorithm correctly identifies this as unphysical, refuses to
    hallucinate negative crosstalk, and safely clamps the unmixing matrix to the Identity
    matrix (0.0 off-diagonals), rather than returning positive coefficients.
    """
    # 1. Setup: Generate Adversarial Anti-Correlated Data
    np.random.seed(42)
    n_pixels = 50_000

    # Create a base signal
    base = np.linspace(0, 4 * np.pi, n_pixels)

    # Signal A is high when Signal B is low, and vice versa
    signal_a = np.sin(base) + 1.5
    signal_b = -np.sin(base) + 1.5

    # Add some noise to avoid degenerate exact-zero entropy cases
    noise_level = 0.2
    signal_a += np.random.normal(0, noise_level, n_pixels)
    signal_b += np.random.normal(0, noise_level, n_pixels)

    # Ensure signals are strictly positive and above background threshold (1e-6)
    signal_a = np.clip(signal_a, 0.1, None)
    signal_b = np.clip(signal_b, 0.1, None)

    sources = np.stack([signal_a, signal_b])

    # Reshape for the API: (Channels, Y, X) -> (2, N, 1)
    adversarial_input = sources.reshape(2, n_pixels, 1)

    # 2. Compute Unmixing Matrix
    # We use all pixels to avoid subsampling noise.
    U_computed = compute_unmixing_matrix(
        adversarial_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=10
    )

    # 3. Assert Physical Constraint (Clamping)
    # The algorithm must not return positive off-diagonals despite the anti-correlation.
    # It should clamp them to 0.0, returning exactly the Identity matrix.

    # Check for strictly non-positive off-diagonals (mathematical safety check)
    off_diagonals = U_computed[~np.eye(2, dtype=bool)]
    assert np.all(off_diagonals <= 0.0), \
        f"Algorithm hallucinated unphysical positive unmixing coefficients: {off_diagonals}"

    # Specifically check that it correctly clamped to the Identity matrix for this adversarial input.
    # We use strict tolerance because the constraint clamping explicitly sets values to 0.0.
    np.testing.assert_allclose(
        U_computed,
        np.eye(2),
        atol=1e-10,
        err_msg="Algorithm failed to clamp unphysical anti-correlations to Identity."
    )
