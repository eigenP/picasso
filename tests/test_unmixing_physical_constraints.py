import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_anticorrelation():
    """
    Testr 🔎: Verify physical constraints (non-negativity of mixing) are enforced.

    This test validates that the algorithm correctly refuses to hallucinate
    unphysical "negative crosstalk" when given strictly anti-correlated signals.

    In fluorescence microscopy, crosstalk is strictly additive: fluorophore A
    may bleed into channel B, meaning $I_B = S_B + \alpha S_A$ where $\\alpha \\ge 0$.
    Consequently, unmixing coefficients must be $\\le 0$ to subtract this crosstalk.

    If two biological structures are mutually exclusive (e.g., nucleus vs. cytoplasm),
    their spatial signals will be strongly anti-correlated. A naive Mutual Information
    minimization might attempt to use a positive coefficient (adding one channel to
    the other) to "fill in the holes" and reduce entropy.

    The algorithm must recognize that this violates the physics of the system,
    clamp the unmixing coefficients to 0, and return the Identity matrix.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # Generate strictly anti-correlated signals
    s1 = np.random.uniform(0.1, 0.9, n_pixels)

    # Add minor noise to avoid degenerate exact-zero entropy bins
    noise = np.random.normal(0, 0.05, n_pixels)
    s2 = 1.0 - s1 + noise

    # Ensure physical positivity of the signals themselves
    s2 = np.clip(s2, 0.01, 1.0)

    # Reshape for the API: (Channels, Pixels, 1)
    image = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # Compute the unmixing matrix
    # quantile=0.0 to use all pixels and ensure stable statistics
    U = compute_unmixing_matrix(
        image,
        verbose=False,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Assert that the algorithm returned the Identity matrix
    # Meaning it refused to perform any unmixing (no off-diagonal elements)
    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-10,
        err_msg="Algorithm hallucinated unphysical crosstalk for anti-correlated signals."
    )
