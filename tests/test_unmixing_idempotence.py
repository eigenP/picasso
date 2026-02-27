import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_idempotence():
    """
    Testr 🔎: Verify the Idempotence invariant of the unmixing algorithm.

    This test validates that:
    Once an image is successfully unmixed into independent sources, passing those
    sources back into the unmixing algorithm yields the Identity matrix (no further updates).

    Mathematically, this represents the property that the unmixed state is a fixed
    point of the algorithm. Unmixing is an idempotent operation:
    U(U(I)) = I (where I is an initially mixed image, and the second application returns the Identity).

    Why this matters:
    - It ensures the algorithm correctly identifies when sources are independent.
    - It proves the algorithm does not "over-unmix" or hallucinate anti-correlations.
    - It verifies the stability of the optimization's stopping criteria.
    """
    np.random.seed(42)

    # 1. Setup: Generate Mixed Synthetic Data
    # 50k pixels provides enough statistical power for stable MI estimation
    n_pixels = 50_000

    # Generate independent sources (Gamma distributed for positivity, common in imaging)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mix the sources to create a correlated signal
    # M represents crosstalk between channels
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape to (Channels, Pixels, 1) for the unmixing API
    mixed_img = mixed_flat.reshape(2, n_pixels, 1)

    # 2. First Pass: Compute Initial Unmixing Matrix
    # We use quantile=0.0 to include all pixels and max_samples=n_pixels
    # to avoid stochastic subsampling noise which could perturb the fixed point.
    U_first = compute_unmixing_matrix(
        mixed_img,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # Verify that the first pass actually did something non-trivial
    assert not np.allclose(U_first, np.eye(2)), \
        "First unmixing matrix should not be Identity for mixed data."

    # 3. Apply Unmixing Matrix to recover Independent Sources
    unmixed_flat = U_first @ mixed_flat
    unmixed_img = unmixed_flat.reshape(2, n_pixels, 1)

    # 4. Second Pass: Verify Idempotence
    # Feed the unmixed image back into the algorithm.
    # Because the signals are now independent, the algorithm should recognize
    # this and return the Identity matrix.
    U_second = compute_unmixing_matrix(
        unmixed_img,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # 5. Assert Idempotence
    # The second unmixing matrix must be Identity within optimization tolerance.
    # A tolerance of 1e-4 allows for tiny numerical float jitter in MI estimation.
    np.testing.assert_allclose(
        U_second,
        np.eye(2),
        atol=1e-4,
        err_msg="Idempotence failed: Algorithm hallucinated further correlations after unmixing."
    )
