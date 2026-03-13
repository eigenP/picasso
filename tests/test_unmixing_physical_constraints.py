import numpy as np
from picasso.unmixing import compute_unmixing_matrix

def test_physical_positivity_constraint():
    """
    Testr 🔎: Verify the strict physical positivity constraint (anti-correlation rejection).

    This test validates that the unmixing algorithm enforces a strict physical bound:
    crosstalk in fluorescence microscopy is strictly additive. Therefore, the unmixing
    process can only subtract signal from other channels (off-diagonal elements MUST be <= 0).

    If the algorithm is presented with anti-correlated signals (where mathematically,
    adding them might reduce mutual information or variance), it must correctly refuse
    to hallucinate unphysical "negative crosstalk". It should clamp the optimization
    and return an Identity matrix, rather than a matrix with positive off-diagonals.

    This ensures the algorithm prioritizes physical reality over pure mathematical optimization.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Generate Anti-Correlated Signals
    # We use a base signal and invert it to create strong anti-correlation.
    # We add noise to avoid degenerate zero-entropy artifacts in binning.
    base_signal = np.random.uniform(0.2, 0.8, n_pixels)

    s1 = base_signal + np.random.normal(0, 0.05, n_pixels)
    s2 = (1.0 - base_signal) + np.random.normal(0, 0.05, n_pixels)

    # Ensure strictly positive values for realistic imaging data
    s1 = np.clip(s1, 0.01, 1.0)
    s2 = np.clip(s2, 0.01, 1.0)

    anti_correlated_input = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Compute Unmixing Matrix
    U = compute_unmixing_matrix(
        anti_correlated_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=20
    )

    # 3. Verify Constraints
    # Off-diagonals must be <= 0. Since it's anti-correlated, math wants > 0.
    # It should be clamped to 0.
    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-7,
        err_msg="Physical constraint failed: Algorithm hallucinated unphysical negative crosstalk (positive off-diagonals) for anti-correlated inputs."
    )
