import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_constraints():
    """
    Testr 🔎: Verify the physical positivity constraint of the unmixing algorithm.

    This test validates that:
    The unmixing algorithm strictly enforces physical constraints (no negative crosstalk)
    even when presented with adversarial inputs that mathematically suggest it.

    In fluorescence imaging, crosstalk is strictly additive (photons cannot cancel each other).
    Therefore, the unmixing coefficients (off-diagonal elements of U) must be <= 0
    to subtract out the crosstalk. If given anti-correlated inputs, a pure mathematical
    optimization might try to "add" one channel to another (positive off-diagonal) to
    minimize Mutual Information.

    The algorithm must recognize this as unphysical, refuse to hallucinate negative
    crosstalk, clamp the coefficients to 0, and safely return an Identity matrix.

    This protects users from generating artifacts when analyzing already-separated
    or actively repelling spatial structures (e.g., mutually exclusive cell compartments).
    """
    np.random.seed(42)

    # 1. Setup: Generate Adversarial Anti-Correlated Data
    # 50k pixels provides enough statistical power
    n_pixels = 50_000

    # Create a base signal to induce strong negative correlation
    base_signal = np.random.uniform(0.2, 0.8, n_pixels)

    # Add independent noise to avoid degenerate exact opposites which might break MI binning
    noise1 = np.random.normal(0, 0.05, n_pixels)
    noise2 = np.random.normal(0, 0.05, n_pixels)

    s1 = base_signal + noise1
    # s2 is strictly anti-correlated with s1
    s2 = (1.0 - base_signal) + noise2

    # Ensure strictly positive values as expected in image data
    s1 = np.clip(s1, 0.01, 1.0)
    s2 = np.clip(s2, 0.01, 1.0)

    # Reshape for API (Channels, Pixels, 1)
    sources = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Compute Unmixing Matrix on Adversarial Data
    u_computed = compute_unmixing_matrix(
        sources,
        verbose=False,
        max_iters=20, # Give it plenty of time to try and fail
        quantile=0.0, # Use all pixels to prevent subsampling from masking the anti-correlation
        max_samples=n_pixels
    )

    # 3. Verify Constraints
    # The algorithm should have clamped any positive updates to 0.0,
    # resulting in the Identity matrix.
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-10,
        err_msg="Physical Constraint failed: Algorithm hallucinated unphysical (positive) crosstalk for anti-correlated inputs."
    )
