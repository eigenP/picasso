import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify Spatial Invariance / "Bag of Pixels" assumption.

    This test validates that the algorithm's output is completely invariant to the
    spatial arrangement of the pixels. Because the objective function (Mutual Information)
    operates entirely on global 1D and 2D marginal histograms of pixel intensities,
    the concept of "spatial neighbor" does not exist in the core optimization.

    Consequently, if we take an N-dimensional image, completely shuffle all its pixels,
    and reshape it into a 1D array, the resulting unmixing matrix must be identically
    the same as the one computed on the original structured image.

    This property ensures the algorithm can naturally process arbitrary spatial dimensions
    (1D, 2D, 3D, temporal) without shape-dependent logic or accidental local operations
    leaking into the calculation.
    """
    # 1. Setup: Generate Synthetic Correlated Data in 3D
    np.random.seed(42)
    shape_3d = (20, 50, 50)  # Z, Y, X
    n_pixels = np.prod(shape_3d)

    # Independent sources
    # We use some structure just so it's not pure noise, although for this invariant
    # pure noise would also work.
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape to 3D image (Channels, Z, Y, X)
    mixed_3d = mixed_flat.reshape(2, *shape_3d)

    # 2. Compute Baseline Unmixing Matrix on Structured 3D Data
    # We set quantile=0.0 and max_samples=n_pixels to guarantee we process exactly
    # all pixels. Subsampling uses strided slicing (np.linspace) which *is* sensitive
    # to data ordering. By processing all pixels, we test the core optimizer's invariance.
    u_base = compute_unmixing_matrix(
        list(mixed_3d),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Spatially Scrambled 1D Data
    # Flatten the spatial dimensions
    mixed_1d = mixed_3d.reshape(2, -1)

    # Generate a random permutation of pixel indices
    perm_indices = np.random.permutation(n_pixels)

    # Apply the same permutation to all channels to maintain cross-channel correlation
    mixed_scrambled = mixed_1d[:, perm_indices]

    # Reshape the scrambled data into a weird arbitrary shape to test dimensionality invariance
    # e.g., (10, 5000)
    mixed_weird_shape = mixed_scrambled.reshape(2, 10, -1)

    # 4. Compute Unmixing Matrix on Scrambled Data
    u_scrambled = compute_unmixing_matrix(
        list(mixed_weird_shape),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Spatial Invariance
    # The unmixing matrix should be strictly identical.
    np.testing.assert_allclose(
        u_scrambled,
        u_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed. Scrambling or reshaping spatial dimensions changed the result."
    )
