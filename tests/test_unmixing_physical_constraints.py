import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_constraints_anti_correlated():
    r"""
    Testr 🔎: Verify physical positivity constraints on unmixing.

    This test validates that:
    The unmixing algorithm enforces a strict physical positivity constraint ($U_{ij} \le 0$).
    Because fluorescence crosstalk is strictly additive, negative crosstalk is physically impossible.
    When given adversarial, anti-correlated inputs (where purely mathematical optimization
    might try to 'add' signals rather than subtract them to minimize MI), the algorithm
    correctly refuses to hallucinate unphysical negative crosstalk, clamps the coefficients
    to 0, and returns an Identity matrix.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # Create anti-correlated signals: when s1 is high, s2 is low.
    base = np.linspace(0, 10, n_pixels)
    # Shuffle base to avoid degenerate binning ordering
    np.random.shuffle(base)

    s1 = base
    s2 = 10.0 - base

    # Add noise to the signals to avoid degenerate cases where Mutual Information
    # optimization fails due to entropy binning artifacts.
    noise1 = np.random.normal(0, 0.5, n_pixels)
    noise2 = np.random.normal(0, 0.5, n_pixels)

    # Ensure strictly positive like fluorescence
    s1 = np.clip(s1 + noise1, 0, None)
    s2 = np.clip(s2 + noise2, 0, None)

    sources = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # Perform Unmixing
    # Use quantile=0.0 and max_samples=n_pixels to avoid subsampling noise
    u_computed = compute_unmixing_matrix(
        sources,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # The algorithm must refuse to unmix anti-correlated signals by returning Identity.
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-4,
        err_msg="Algorithm hallucinated unphysical negative crosstalk for anti-correlated inputs."
    )
