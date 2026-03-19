import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_unphysical_anticorrelation():
    """
    Testr 🔎: Verify physical bound constraints against adversarial inputs.

    This test validates that the algorithm enforces strict physical positivity
    constraints ($U_{ij} \\le 0$) when faced with adversarial, unphysical inputs.

    In fluorescence microscopy, crosstalk is strictly additive. Thus, valid
    unmixing matrices can only subtract signals (off-diagonals <= 0). If given
    strictly anti-correlated inputs, a pure mathematical optimization (minimizing
    Mutual Information) might try to *add* one signal to the other to destroy
    the anti-correlation, resulting in a positive off-diagonal coefficient.

    This test provides explicitly anti-correlated signals and verifies that the
    algorithm safely rejects the unphysical optimization, clamps the coefficients,
    and returns the Identity matrix (no unmixing).
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Generate Anti-Correlated Signals
    # We create a base signal and its inverse
    base_signal = np.random.uniform(10, 100, n_pixels)

    s1 = base_signal
    # Channel 2 is the exact opposite, ensuring strict anti-correlation
    s2 = 110 - base_signal

    # Add some independent noise to avoid degenerate cases where entropy
    # binning artifacts might cause MI optimization to fail trivially.
    noise1 = np.random.normal(0, 1, n_pixels)
    noise2 = np.random.normal(0, 1, n_pixels)

    s1 = s1 + noise1
    s2 = s2 + noise2

    # Reshape for API (Channels, Pixels, 1)
    adversarial_input = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Compute Unmixing Matrix
    u_computed = compute_unmixing_matrix(
        adversarial_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # 3. Assert Physical Constraint is Enforced
    # The algorithm should recognize the mathematical optimization leads to
    # positive off-diagonals and clamp them to 0, resulting in the Identity matrix.
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-7,
        err_msg="Algorithm hallucinated unphysical (positive) crosstalk for anti-correlated inputs."
    )
