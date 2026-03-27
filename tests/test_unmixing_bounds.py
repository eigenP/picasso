import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_anticorrelated_signals():
    """
    Testr 🔎: Verify the physical positivity constraint on unmixing bounds.

    This test validates that:
    1. The unmixing algorithm safely rejects mutually exclusive or strictly
       anti-correlated inputs.
    2. The unmixing coefficients are strictly clamped to <= 0 ($U_{ij} \\le 0$).

    Why this matters:
    Fluorescence signals are fundamentally additive. Signal crosstalk implies that
    a photon intended for Channel 1 was detected in Channel 2. Therefore, unmixing
    must only *subtract* a scaled version of one channel from another ($Y - \\alpha X$).
    If signals are strictly anti-correlated (e.g. they represent mutually exclusive
    cellular structures), a naive Mutual Information optimizer might try to add them
    together ($\\alpha < 0$, meaning $U_{ij} > 0$) to reduce the total entropy.
    This is mathematically optimal but physically impossible, resulting in "hallucinated"
    crosstalk or over-unmixing. The algorithm MUST refuse this and return the Identity matrix.
    """
    np.random.seed(42)

    n_pixels = 50_000

    # 1. Create Adversarial Anti-Correlated Data
    # Base signal
    base_signal = np.random.uniform(0.2, 0.8, n_pixels)

    # Make them strictly anti-correlated
    # If s1 is high, s2 is low
    s1 = base_signal
    s2 = 1.0 - base_signal

    # Add a small amount of independent noise to avoid degenerate exact 0-entropy binning cases
    noise_level = 0.05
    s1 += np.random.normal(0, noise_level, n_pixels)
    s2 += np.random.normal(0, noise_level, n_pixels)

    # Ensure all values remain strictly positive (like actual fluorescence)
    s1 = np.clip(s1, 0.01, None)
    s2 = np.clip(s2, 0.01, None)

    sources = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Compute Unmixing Matrix
    # We use all pixels to verify the mathematical behavior precisely
    u_computed = compute_unmixing_matrix(
        list(sources),
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # 3. Assert Rejection of Unphysical Mathematical Optima
    # The pure MI minimum would likely try to add the signals to flatten the distribution.
    # The physical constraint must clip those updates to 0, resulting in the Identity matrix.
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-10,
        err_msg="Algorithm hallucinated unphysical negative crosstalk (U_ij > 0) on anti-correlated data."
    )

    # Explicitly check bounds
    assert np.all(u_computed[~np.eye(2, dtype=bool)] <= 0.0), \
        "Unmixing coefficients violated physical positivity bounds (must be <= 0)."
