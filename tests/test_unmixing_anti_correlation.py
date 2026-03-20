import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_anti_correlation():
    """
    Testr 🔎: Verify the physical non-negativity constraint via adversarial anti-correlated inputs.

    This test validates that:
    When presented with strictly anti-correlated signals, the algorithm correctly
    refuses to "unmix" them using unphysical positive crosstalk coefficients.

    Why this matters:
    Fluorescence crosstalk is strictly additive (positive mixing). Therefore, the unmixing
    matrix off-diagonals must be strictly non-positive (U_ij <= 0). If the algorithm
    encounters anti-correlated signals (which could happen due to noise or spatial exclusion
    in biology), pure Mutual Information minimization would try to use positive coefficients
    to decorrelate them. The algorithm must safely hit its constraint bounds and return the Identity matrix.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # Generate a base signal
    base_signal = np.random.uniform(0.1, 1.0, n_pixels)

    # Create an anti-correlated signal
    anti_signal = 1.1 - base_signal

    # Add some independent noise to avoid degenerate zero-entropy states
    # where all points fall into exactly a 1D line in the 2D histogram
    noise1 = np.random.normal(0, 0.05, n_pixels)
    noise2 = np.random.normal(0, 0.05, n_pixels)

    s1 = np.clip(base_signal + noise1, 0.01, None)
    s2 = np.clip(anti_signal + noise2, 0.01, None)

    mixed_input = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # Compute unmixing matrix
    U = compute_unmixing_matrix(
        mixed_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # The pure mathematical optimum for anti-correlated signals would involve positive off-diagonals.
    # We assert that the physical bounds correctly clamped them to 0 (Identity matrix).
    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-5,
        err_msg="Algorithm hallucinated unphysical negative crosstalk for anti-correlated inputs."
    )
