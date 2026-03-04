import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_anti_correlation_rejection():
    """
    Testr 🔎: Verify the physical positivity constraint of the unmixing algorithm.

    This test validates that:
    The algorithm refuses to hallucinate "negative crosstalk" when presented with
    anti-correlated signals.

    Why this matters:
    Fluorescence crosstalk strictly adds photons to adjacent channels; it never
    subtracts them. Therefore, an unmixing matrix must only subtract signals
    (represented by non-positive off-diagonal elements in our convention, where
    the unmixed signal is U @ M, and U has 1s on the diagonal).

    If the algorithm encounters signals that are strongly anti-correlated (i.e.,
    when one is high, the other is low), an unconstrained Mutual Information
    optimizer might attempt to add the signals together to minimize MI.
    This test verifies that the `compute_unmixing_matrix` function correctly
    clamps unphysical positive updates to 0.0, returning the Identity matrix
    for such adversarial inputs.
    """
    np.random.seed(42)

    # 1. Setup: Generate Synthetic Anti-Correlated Data
    # 50k pixels provides enough statistical power for stable MI estimation
    n_pixels = 50_000

    # Generate a strong primary source
    # Gamma distribution simulates a long-tailed fluorescence signal
    s1 = np.random.gamma(2, 2, n_pixels)

    # Create an anti-correlated source:
    # We subtract a fraction of s1 from another independent source, ensuring it stays positive
    s2_base = np.random.gamma(5, 2, n_pixels)
    # The resulting s2 will decrease when s1 increases
    s2 = np.maximum(s2_base - 0.5 * s1, 0.1)  # Clamp to small positive value

    # Shape: (Channels, Pixels, 1) for the function
    sources = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Perform Unmixing
    # Use quantile=0.0 and max_samples=n_pixels to avoid subsampling stochasticity
    # We want to test the core optimization constraint directly.
    u_computed = compute_unmixing_matrix(
        sources,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=20
    )

    # 3. Assert Positivity Constraint (Anti-Correlation Rejection)
    # Because the signals are anti-correlated, the unconstrained optimal 'alpha'
    # would be positive (meaning off-diagonal elements of U would be positive).
    # The algorithm must recognize this is unphysical and clamp it to 0.
    # Therefore, the returned matrix should be exactly the Identity matrix.

    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-10,
        err_msg=(
            "Physical Constraint Failed: Algorithm applied unphysical updates "
            "(positive off-diagonals) to anti-correlated signals.\n"
            f"Computed Matrix:\n{u_computed}"
        )
    )
