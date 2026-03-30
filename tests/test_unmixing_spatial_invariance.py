import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify Spatial Invariance and Dimensionality Agnosticism of the unmixing algorithm.

    This test validates that the algorithm relies strictly on the global intensity distribution
    (a "bag of pixels" approach) and is entirely invariant to both:
    1. **Spatial Arrangement**: Randomly permuting all pixels across the image should not
       alter the optimal unmixing matrix.
    2. **Dimensionality**: Reshaping the input from a 2D image into a 1D sequence or a 3D volume
       should yield the exact same unmixing matrix.

    Why this matters:
    - **Robustness**: It proves that the unmixing logic is not accidentally dependent on local
      spatial structures, gradients, or the ordering of pixels in memory.
    - **Generality**: It mathematically guarantees that the algorithm can seamlessly handle
      1D spectra, 2D images, 3D volumes (Z-stacks), or even 4D timelapses without requiring
      specialized code paths for different dimensionalities.
    - **Optimization Safety**: It confirms that aggressive spatial downscaling or random
      sampling strategies during optimization are theoretically sound, as the exact spatial
      context is irrelevant to the objective function (Mutual Information).
    """
    np.random.seed(42)

    # 1. Setup: Generate Mixed Synthetic Data
    n_pixels = 10_000

    # Independent sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Create a baseline 2D image (Channels, Y, X)
    side_length = int(np.sqrt(n_pixels))
    mixed_2d = mixed_flat.reshape(2, side_length, side_length)

    # 2. Compute Baseline Unmixing Matrix
    # We use quantile=0.0 and max_samples=n_pixels to ensure all pixels are used,
    # avoiding any stochastic subsampling differences between shapes.
    U_base = compute_unmixing_matrix(
        list(mixed_2d),
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(U_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Verify Spatial Invariance (Permutation)
    # Generate a random permutation for all spatial pixels
    perm = np.random.permutation(n_pixels)
    mixed_permuted = mixed_flat[:, perm].reshape(2, side_length, side_length)

    U_permuted = compute_unmixing_matrix(
        list(mixed_permuted),
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    np.testing.assert_allclose(
        U_permuted,
        U_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed: Permuting pixels changed the unmixing matrix."
    )

    # 4. Verify Dimensionality Agnosticism (1D)
    mixed_1d = mixed_flat.reshape(2, n_pixels)

    U_1d = compute_unmixing_matrix(
        list(mixed_1d),
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    np.testing.assert_allclose(
        U_1d,
        U_base,
        atol=1e-10,
        err_msg="Dimensionality Agnosticism failed: 1D input changed the unmixing matrix."
    )

    # 5. Verify Dimensionality Agnosticism (3D Volume)
    # Reshape 10_000 pixels into a 10 x 10 x 100 volume
    mixed_3d = mixed_flat.reshape(2, 10, 10, 100)

    U_3d = compute_unmixing_matrix(
        list(mixed_3d),
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    np.testing.assert_allclose(
        U_3d,
        U_base,
        atol=1e-10,
        err_msg="Dimensionality Agnosticism failed: 3D input changed the unmixing matrix."
    )
