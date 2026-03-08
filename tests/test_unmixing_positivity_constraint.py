import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_positivity_constraint_anti_correlated():
    """
    Testr 🔎: Verify the Positivity Constraint (No Negative Crosstalk).

    This test validates that the algorithm enforces a strict physical
    positivity constraint on crosstalk. In fluorescence microscopy,
    crosstalk is strictly additive (photons from fluorophore A bleed into
    channel B). Therefore, the unmixing coefficients (which subtract crosstalk)
    must be strictly non-positive (U_ij <= 0).

    When given strictly anti-correlated inputs (where channel A is high when
    channel B is low), a purely mathematical blind source separation algorithm
    might attempt to minimize Mutual Information by adding the channels together
    (hallucinating negative crosstalk, resulting in positive unmixing
    coefficients U_ij > 0).

    This test ensures that the physical constraint overrides the mathematical
    optimization, clamping the unphysical coefficients to 0 and correctly
    returning the Identity matrix (refusing to unmix when only unphysical
    "crosstalk" would decrease Mutual Information).
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Generate strictly anti-correlated signals
    # If base_signal is high, s1 is high, s2 is low.
    base_signal = np.random.uniform(2, 10, n_pixels)
    s1 = base_signal
    s2 = 12.0 - base_signal  # Anti-correlated

    # 2. Add independent noise
    # We add independent noise to avoid degenerate cases where the signals
    # perfectly sum to a constant. Perfect constants have 0 entropy, which
    # can cause division-by-zero or binning artifacts in the MI estimator
    # (as noted in Testr's memory guidelines).
    noise1 = np.random.normal(0, 0.5, n_pixels)
    noise2 = np.random.normal(0, 0.5, n_pixels)

    s1 = np.clip(s1 + noise1, 0.1, None)
    s2 = np.clip(s2 + noise2, 0.1, None)

    # 3. Format as image input (Channels, Pixels, 1)
    anti_correlated_input = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 4. Compute Unmixing Matrix
    # A naive unconstrained MI minimization would produce positive off-diagonals.
    u_computed = compute_unmixing_matrix(
        anti_correlated_input,
        verbose=False,
        max_iters=20,
        quantile=0.0, # Use all pixels to avoid selection bias
        max_samples=n_pixels
    )

    # 5. Assert the result is strictly clamped to the Identity matrix
    # The algorithm must recognize the unphysical request and refuse to unmix.
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-10,
        err_msg="Positivity Constraint failed: Algorithm hallucinated unphysical negative crosstalk on anti-correlated data."
    )
