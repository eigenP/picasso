import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance of the unmixing algorithm.

    This test validates that the algorithm's unmixing optimization is strictly
    spatially invariant. The unmixing process treats the image as a global "bag of pixels",
    meaning the physical arrangement or spatial dimensionality (1D, 2D, 3D, temporal)
    of the input pixels must not affect the resulting unmixing matrix.

    Why this matters:
    - It guarantees that the core algorithm is safely abstracted from spatial dimensions,
      allowing it to generalize trivially from 2D images to 3D volumes or n-D hyperstacks.
    - It confirms that local pixel context is not implicitly relied upon by the optimization,
      meaning arbitrary spatial shuffling or slicing will not alter the statistical behavior.

    We test this by computing the unmixing matrix on a structured 2D image, then completely
    scrambling the spatial positions of all pixels, and verifying the exact same unmixing
    matrix is computed.
    """
    np.random.seed(42)

    # 1. Setup: Generate structured synthetic 2D data
    # Create somewhat structured sources (e.g., gradients or blocks) to simulate spatial meaning
    width, height = 200, 200
    n_pixels = width * height

    # Source 1: vertical gradient with noise
    s1_base = np.linspace(0, 10, width).reshape(1, width)
    s1 = np.repeat(s1_base, height, axis=0) + np.random.gamma(1, 1, (height, width))

    # Source 2: horizontal gradient with noise
    s2_base = np.linspace(0, 10, height).reshape(height, 1)
    s2 = np.repeat(s2_base, width, axis=1) + np.random.gamma(1, 1, (height, width))

    sources = np.stack([s1, s2]) # Shape: (2, 200, 200)

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])

    # Mix sources
    # Reshape for tensordot
    sources_flat = sources.reshape(2, -1)
    mixed_flat = M @ sources_flat

    # Reshape back to 2D structured image
    mixed_image = mixed_flat.reshape(2, height, width)

    # 2. Compute Baseline Unmixing Matrix
    # Use quantile=0.0 and max_samples=n_pixels to ensure all pixels are evaluated,
    # thereby avoiding subsampling artifacts which could depend on flat memory layout.
    u_base = compute_unmixing_matrix(
        list(mixed_image),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 3. Scramble Spatial Information
    # Generate a random permutation of pixel indices
    perm = np.random.permutation(n_pixels)

    # Flatten the channels, shuffle, and reshape back to a 2D image (destroying spatial context)
    mixed_scrambled = np.zeros_like(mixed_image)
    for c in range(2):
        flat_c = mixed_image[c].flatten()
        scrambled_c = flat_c[perm]
        mixed_scrambled[c] = scrambled_c.reshape(height, width)

    # 4. Compute Unmixing Matrix on Scrambled Data
    u_scrambled = compute_unmixing_matrix(
        list(mixed_scrambled),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Spatial Invariance
    # Since the global histogram of values is strictly identical, the Mutual Information
    # minimization should return the exact same matrix.
    np.testing.assert_allclose(
        u_scrambled,
        u_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed. Scrambling pixels changed the unmixing matrix."
    )
