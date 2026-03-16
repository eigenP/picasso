import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_unphysical_crosstalk():
    """
    Testr 🔎: Verify physical constraint enforcement against unphysical inputs.

    This test validates that:
    When presented with strongly anti-correlated input signals—which would mathematically
    suggest "negative crosstalk"—the algorithm correctly refuses to unmix them.
    It should safely clamp the unmixing coefficients and return the Identity matrix.

    Why this matters:
    - In fluorescence microscopy, crosstalk is strictly *additive* (photons from fluorophore A
      bleed into the detection channel for fluorophore B).
    - It is physically impossible to have "negative photons" or "negative bleed-through".
    - A naive mathematical optimization of Mutual Information might attempt to subtract a
      negative amount (i.e., add channels together) to minimize correlations, leading to
      wildly incorrect, unphysical results that amplify noise and distort data.
    - This test ensures the algorithm's hard non-positivity constraint on off-diagonal
      unmixing coefficients ($U_{ij} \\le 0$) correctly overrides pure mathematical optimization.
    """
    np.random.seed(42)

    # 1. Setup: Generate Synthetic Independent Sources
    # We use a large number of pixels for stable statistical estimation.
    # We also add noise to the signals. Without noise, strictly negative mixing
    # can create degenerate 1D manifolds that cause entropy binning artifacts.
    n_pixels = 50_000

    # Gamma distribution provides positive, skewed signals typical of fluorescence
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # 2. Create Unphysical "Anti-Correlated" Signal
    # We mix the sources with a matrix containing *negative* off-diagonals.
    # Mathematically, $M = [[1, -0.6], [-0.6, 1]]$.
    # This simulates a scenario where high signal in Channel 1 suppresses Channel 2.
    # While unphysical for additive bleed-through, it aggressively tests the bounds.
    M_unphysical = np.array([[1.0, -0.6], [-0.6, 1.0]])
    mixed_flat = M_unphysical @ sources

    # Because our mixing matrix has large negative values, some pixel values might drop < 0.
    # We clip them to a small positive baseline to simulate camera offset and keep them valid
    # for the algorithm's background filter (> 1e-6).
    mixed_flat = np.clip(mixed_flat, 0.01, None)

    # Add some independent Gaussian noise to ensure the distributions remain 2D
    # and don't collapse perfectly into lines, which can break simple histogram entropy.
    noise = np.random.normal(0, 0.1, mixed_flat.shape)
    mixed_flat = np.clip(mixed_flat + noise, 0.01, None)

    # Reshape for API (Channels, Pixels, 1)
    mixed_input = mixed_flat.reshape(2, n_pixels, 1)

    # 3. Perform Unmixing
    # We use quantile=0.0 to use all data and disable subsampling noise.
    # We expect the algorithm to *try* to optimize, find that only positive
    # off-diagonal unmixing coefficients (which correspond to negative crosstalk)
    # would help, and then clamp them to 0.0.
    computed_U = compute_unmixing_matrix(
        mixed_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=10 # It should "converge" immediately by hitting the bounds
    )

    # 4. Verification
    # The resulting matrix MUST be the Identity matrix.
    # This means the algorithm explicitly refused to do any unmixing.
    # We allow a tiny tolerance for numerical jitter from the first optimization step
    # before clamping takes effect.
    np.testing.assert_allclose(
        computed_U,
        np.eye(2),
        atol=1e-3,
        err_msg="Algorithm hallucinated unphysical (positive) unmixing coefficients for anti-correlated inputs."
    )

    # Explicitly check the clamping logic bounds.
    # Off-diagonals must be <= 0.0.
    # (Because U_ij represents the *negative* of the mixing, an unphysical negative
    # mixing would require a positive U_ij to correct it. The algorithm enforces U_ij <= 0).
    off_diagonals = computed_U[~np.eye(2, dtype=bool)]
    assert np.all(off_diagonals <= 0.0), \
        f"Algorithm violated physical non-positivity constraint. Off-diagonals: {off_diagonals}"
    assert np.all(off_diagonals >= -1e-3), \
        f"Algorithm unmixed in the wrong direction despite anti-correlation. Off-diagonals: {off_diagonals}"
