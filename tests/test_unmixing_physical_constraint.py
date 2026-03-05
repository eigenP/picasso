import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_positivity_constraint():
    """
    Testr 🔎: Verify the physical positivity constraint of the unmixing algorithm.

    This test validates that:
    The unmixing algorithm enforces a strict physical constraint that fluorescence
    crosstalk is strictly additive. In mathematical terms, the unmixing matrix $U$
    must have non-positive off-diagonal elements ($U_{ij} \le 0$).

    When provided with anti-correlated inputs (which would theoretically require
    "negative crosstalk" or a positive off-diagonal element to unmix), the algorithm
    correctly refuses to hallucinate unphysical correlations, clamping the coefficients
    to 0 and returning an Identity matrix.

    Why this matters:
    - Fluorescence cannot be negative. Crosstalk only adds signal, never subtracts.
    - An algorithm without this constraint might overfit to noise or preprocessing
      artifacts (like improper background subtraction) that create artificial
      anti-correlations, leading to unphysical and corrupt unmixing results.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Generate Anti-Correlated Synthetic Data
    # A simple way to generate anti-correlated data is to have one source go up
    # while the other goes down.
    base = np.random.uniform(0, 10, n_pixels)

    # Source 1 is just the base
    s1 = base
    # Source 2 is inverted (anti-correlated)
    s2 = 10.0 - base

    # Add some independent noise to avoid perfectly degenerate data
    s1 = s1 + np.random.normal(0, 0.5, n_pixels)
    s2 = s2 + np.random.normal(0, 0.5, n_pixels)

    # Ensure strictly positive (like light)
    s1 = np.clip(s1, 0.1, None)
    s2 = np.clip(s2, 0.1, None)

    sources = np.stack([s1, s2])

    # Reshape for the API (Channels, Pixels, 1)
    mixed_input = sources.reshape(2, n_pixels, 1)

    # 2. Compute Unmixing Matrix
    u_computed = compute_unmixing_matrix(
        mixed_input,
        verbose=False,
        max_iters=50,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 3. Verify the Physical Constraint
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-4,
        err_msg="Algorithm hallucinated unphysical (positive) crosstalk for anti-correlated inputs."
    )
