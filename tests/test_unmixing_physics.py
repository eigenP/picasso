import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_positivity_constraint():
    """
    Testr 🔎: Verify the physical positivity constraint of the unmixing algorithm.

    This test validates that:
    The unmixing algorithm enforces a strict physical positivity constraint ($U_{ij} \\le 0$).
    Because fluorescence crosstalk is strictly additive (you cannot have "negative photons"),
    the mixing matrix $M$ must have non-negative entries, meaning the unmixing matrix $U \approx M^{-1}$
    (when normalized to 1 on the diagonal) must only subtract signal (off-diagonals $\\le 0$).

    If the algorithm is fed adversarial, strictly anti-correlated signals (which mathematically
    could reduce Mutual Information by adding them together, i.e., $U_{ij} > 0$), it must
    correctly refuse to hallucinate unphysical negative crosstalk. It should clamp the coefficients
    to 0 and return an Identity matrix.

    Why this matters:
    - Pure mathematical optimization (minimizing MI) without physical bounds would happily
      add anti-correlated channels together to reduce entropy.
    - Failing to clamp these values results in "over-unmixing" and hallucinating artifacts.
    - This explicitly tests the `mat_cumul[row, col] = 0.0` clipping logic in the algorithm.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Setup: Generate Adversarial Anti-Correlated Data
    # Base signal
    s1 = np.random.uniform(0.1, 1.0, n_pixels)

    # Create a strongly anti-correlated signal.
    # When s1 is high, s2 is low.
    s2 = 1.1 - 0.5 * s1

    # Add a little noise to avoid degenerate 1D histograms
    # which can cause fast-histogram to behave weirdly if bins are perfectly empty
    s2 += np.random.normal(0, 0.05, n_pixels)

    # Stack into expected API format (Channels, Pixels, 1)
    mixed = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Run Unmixing Algorithm
    # Use quantile=0.0 and max_samples=n_pixels to avoid subsampling noise.
    # Max iterations can be small since it should clamp immediately.
    U = compute_unmixing_matrix(
        mixed,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=10
    )

    # 3. Assert Physical Constraint (Identity Matrix)
    # The algorithm should have attempted to use a positive alpha to minimize MI,
    # but the physical constraint logic must have clamped it to 0.0.
    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-8,
        err_msg="Algorithm failed physical positivity constraint. It hallucinated unphysical crosstalk (U != Identity)."
    )
