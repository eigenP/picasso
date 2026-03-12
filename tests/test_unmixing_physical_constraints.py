import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_physical_positivity_constraint():
    """
    Testr 🔎: Verify the physical positivity constraint of the unmixing algorithm.

    This test validates that the algorithm correctly enforces a strict physical
    positivity constraint (or non-negative crosstalk) when encountering adversarial inputs.

    In fluorescence microscopy, crosstalk is strictly additive (signal from one channel
    bleeds into another). Therefore, the unmixing coefficients must be non-positive to
    subtract this crosstalk. If the algorithm is presented with strictly anti-correlated
    signals (where high signal in one channel means low signal in the other), a pure
    mathematical optimization without constraints might try to "unmix" them by
    hallucinating positive crosstalk (adding signal to reduce mutual information).

    The algorithm must recognize this as unphysical, safely reject it by clamping the
    coefficients to 0, and return the Identity matrix.

    Why this matters:
    - Verifies that the optimization is constrained by physical reality, not just math.
    - Prevents the hallucination of non-existent signals (negative concentrations).
    - Ensures robustness against unexpected or adversarial input distributions.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Generate strictly anti-correlated data: if x is high, y is low
    # This acts as an adversarial input that would tempt an unconstrained optimizer
    # to find a large positive coefficient.
    x = np.random.uniform(10, 100, n_pixels)
    y = 110 - x

    # 2. Add noise to avoid degenerate cases
    # Pure deterministic relationships can cause entropy binning artifacts
    # where all values fall exactly into a single diagonal line of bins.
    x += np.random.normal(0, 5, n_pixels)
    y += np.random.normal(0, 5, n_pixels)

    # 3. Ensure positivity for the algorithm
    # The algorithm operates on intensities, so we clamp to avoid non-physical negative values
    x = np.clip(x, 1, None)
    y = np.clip(y, 1, None)

    sources = np.stack([x, y]).reshape(2, n_pixels, 1)

    # 4. Compute unmixing matrix
    # Use quantile=0.0 and max_samples=n_pixels to avoid subsampling noise
    U = compute_unmixing_matrix(
        sources,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # 5. Assert that the unmixing matrix is Identity
    # It must refuse to unmix anti-correlated signals
    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-4,
        err_msg="Algorithm failed to reject anti-correlated signals. Unphysical negative crosstalk (positive coefficients) hallucinated."
    )
