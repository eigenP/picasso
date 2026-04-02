import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the exact Spatial Invariance (Bag-of-Pixels) property.

    This test validates that the unmixing algorithm is completely independent
    of spatial organization. Because it optimizes the joint distribution
    (Mutual Information) via a global histogram, it must be strictly
    spatially invariant.

    Specifically, we verify two properties:
    1. **Dimensional Agnosticism**: The algorithm yields the identical matrix
       whether the input is 1D, 2D, or 3D, as long as the pixel values are the same.
    2. **Permutation Invariance**: Shuffling the pixels randomly (but consistently
       across all channels) does not change the unmixing matrix at all.

    Why this matters:
    This invariant proves that the algorithm successfully treats the data as a
    "bag of pixels" and guarantees that users can feed arbitrary dimensional
    time-series (1D), standard images (2D), or volumetric data (3D) without
    needing to change the underlying algorithm or expecting different behavior.
    """
    np.random.seed(42)
    n_pixels = 24_000 # Highly composite number

    # 1. Setup: Generate Synthetic Correlated Data
    # Independent sources
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.3, 1.0]])
    mixed_flat = M @ sources

    # 2. Reshape into different dimensionalities
    # 1D format
    mixed_1d = [mixed_flat[0], mixed_flat[1]]

    # 2D format (e.g. 120 x 200)
    mixed_2d = [mixed_flat[0].reshape(120, 200), mixed_flat[1].reshape(120, 200)]

    # 3D format (e.g. 20 x 30 x 40)
    mixed_3d = [mixed_flat[0].reshape(20, 30, 40), mixed_flat[1].reshape(20, 30, 40)]

    # 3. Compute Baseline Unmixing Matrix (1D)
    # Use quantile=0.0 and max_samples=n_pixels to ensure all pixels are used
    # and to avoid any stochastic sub-sampling differences based on shapes.
    u_base_1d = compute_unmixing_matrix(
        mixed_1d,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base_1d, np.eye(2)), \
        "Baseline unmixing matrix is trivial. Need correlated data."

    # 4. Verify Dimensional Agnosticism
    u_2d = compute_unmixing_matrix(
        mixed_2d,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    np.testing.assert_allclose(
        u_2d, u_base_1d, atol=1e-10,
        err_msg="Spatial Invariance failed: 2D matrix differs from 1D."
    )

    u_3d = compute_unmixing_matrix(
        mixed_3d,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    np.testing.assert_allclose(
        u_3d, u_base_1d, atol=1e-10,
        err_msg="Spatial Invariance failed: 3D matrix differs from 1D."
    )

    # 5. Verify Permutation Invariance
    # Shuffle pixels consistently across channels
    shuffle_idx = np.random.permutation(n_pixels)
    mixed_shuffled = [
        mixed_flat[0][shuffle_idx],
        mixed_flat[1][shuffle_idx]
    ]

    u_shuffled = compute_unmixing_matrix(
        mixed_shuffled,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    np.testing.assert_allclose(
        u_shuffled, u_base_1d, atol=1e-10,
        err_msg="Spatial Invariance failed: Shuffling pixels changed the matrix."
    )
