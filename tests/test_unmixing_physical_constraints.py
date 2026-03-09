import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_anti_correlation():
    """
    Testr 🔎: Verify the physical positivity constraint of the unmixing algorithm.

    This test validates that the algorithm correctly enforces a strict physical
    constraint: fluorescence crosstalk is purely additive. Therefore, the unmixing
    matrix off-diagonals must be non-positive ($U_{ij} \\le 0$).

    When provided with strictly anti-correlated inputs (an unphysical scenario for
    crosstalk), purely mathematical optimization of Mutual Information might attempt
    to hallucinate a negative crosstalk (resulting in positive off-diagonal entries in U)
    to minimize MI.

    We explicitly test this adversarial case to verify that the domain-specific
    physical constraint overrides pure mathematical optimization, clamping the
    coefficients and returning an Identity matrix (refusing to unmix).
    """
    np.random.seed(42)
    n_pixels = 50_000

    # Create strongly anti-correlated signal
    x = np.random.uniform(10, 100, n_pixels)
    y = 110 - x

    # Add noise to avoid degenerate cases where MI estimation fails due to exact binning artifacts
    noise_level = 5.0
    x += np.random.normal(0, noise_level, n_pixels)
    y += np.random.normal(0, noise_level, n_pixels)

    # Ensure strictly positive values as expected in fluorescence
    x = np.clip(x, 1, None)
    y = np.clip(y, 1, None)

    sources = np.stack([x, y])
    mixed_input = sources.reshape(2, n_pixels, 1)

    u_computed = compute_unmixing_matrix(
        mixed_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=20
    )

    # The algorithm should completely reject the unphysical anti-correlation,
    # clamping any positive off-diagonal updates to 0.0, effectively returning
    # the Identity matrix.
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-10,
        err_msg="Physical constraint failed: Algorithm hallucinated unphysical negative crosstalk for anti-correlated inputs."
    )
