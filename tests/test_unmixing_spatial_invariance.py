import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance of the unmixing algorithm.

    This test validates that:
    The unmixing algorithm is strictly a "bag of pixels" approach. Randomly permuting
    the spatial arrangement of pixels yields the exact same unmixing matrix.
    Because the algorithm only relies on the global joint distribution of pixel
    intensities (via Mutual Information), spatial context is irrelevant.

    Why this matters:
    - It confirms that the algorithm can naturally support arbitrary spatial
      dimensions (1D, 2D, 3D, temporal) without relying on spatial features.
    - It ensures no accidental coupling to spatial structures or convolution-like
      artifacts inside the core optimization.
    """
    np.random.seed(42)

    # 1. Setup: Generate Synthetic Correlated Data (2D Images)
    height, width = 100, 100
    n_pixels = height * width

    # Independent sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape to (Channels, Height, Width) for the API
    mixed_image = mixed_flat.reshape(2, height, width)

    # 2. Compute Baseline Unmixing Matrix
    # We must use quantile=0.0 and max_samples=n_pixels to bypass subsampling.
    # Deterministic subsampling depends on the array order. If we shuffle the
    # pixels, the strided slice would pick different subsets, breaking exact equality.
    u_base = compute_unmixing_matrix(
        list(mixed_image),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Spatially Permuted Data
    # Generate a random permutation for all pixels
    permutation = np.random.permutation(n_pixels)

    # We must permute both channels with the exact same spatial permutation
    # to maintain their pixel-wise correlations.
    mixed_permuted_flat = mixed_flat[:, permutation]

    # Reshape back to the original image dimensions
    mixed_permuted_image = mixed_permuted_flat.reshape(2, height, width)

    # 4. Compute Unmixing Matrix on Permuted Data
    u_permuted = compute_unmixing_matrix(
        list(mixed_permuted_image),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Spatial Invariance
    # The unmixing matrix should be strictly identical.
    np.testing.assert_allclose(
        u_permuted,
        u_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed. Permuting pixels changed the unmixing matrix."
    )
