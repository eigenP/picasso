import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance (Bag-of-Pixels) property of the unmixing algorithm.

    This test validates that:
    1. The unmixing algorithm is completely agnostic to the spatial dimensions of the input.
       Whether the image is provided as 1D, 2D, or 3D, the resulting unmixing matrix
       must be exactly identical.
    2. The spatial arrangement of the pixels does not matter. Randomly shuffling
       the pixels across all channels identically must yield the exact same matrix.

    Why this matters:
    - The algorithm operates on global intensity distributions (histograms) and mutual information.
      It does not use local neighborhood context.
    - Confirming this invariant guarantees that the tool safely supports any arbitrary
      input shape (Z-stacks, temporal data, standard 2D images, flattened arrays)
      without silently producing varying analytical results.
    - If this test fails, it suggests an implicit spatial bias or an unintended order-dependent
      logic flaw (like an accidental running state or unhandled shape in the API).
    """
    np.random.seed(42)

    # 1. Setup: Generate Synthetic Correlated Data
    # 10,000 pixels is enough for stable histograms and easily reshaping into 100x100 and 10x10x100
    n_pixels = 10_000

    # Independent sources (Gamma distributed for positivity, common in imaging)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources  # Shape: (2, 10000)

    # 2. Create different spatial representations
    # 1D Array
    mixed_1d = mixed_flat.copy()

    # 2D Image
    mixed_2d = mixed_flat.reshape(2, 100, 100)

    # 3D Stack
    mixed_3d = mixed_flat.reshape(2, 10, 10, 100)

    # Shuffled 1D (to prove order independence)
    shuffle_idx = np.random.permutation(n_pixels)
    mixed_shuffled = mixed_flat[:, shuffle_idx]

    # 3. Compute Unmixing Matrices
    # Use quantile=0.0 to use all pixels and ensure strict determinism (no thresholding changes from spatial shapes).
    # Use max_samples=n_pixels to avoid subsampling randomness.
    kwargs = dict(
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    u_1d = compute_unmixing_matrix(list(mixed_1d), **kwargs)

    # Assert non-trivial matrix
    assert not np.allclose(u_1d, np.eye(2)), "Baseline unmixing matrix is trivial. Use more correlated data."

    u_2d = compute_unmixing_matrix(list(mixed_2d), **kwargs)
    u_3d = compute_unmixing_matrix(list(mixed_3d), **kwargs)
    u_shuffled = compute_unmixing_matrix(list(mixed_shuffled), **kwargs)

    # 4. Assert Exact Invariance
    # Because this is a strict analytical property of the data representation and the algorithm
    # mathematically ignores position, the results must be identical to numerical precision.

    np.testing.assert_allclose(
        u_2d,
        u_1d,
        atol=1e-10,
        err_msg="Spatial Invariance failed: 2D reshape altered the unmixing matrix."
    )

    np.testing.assert_allclose(
        u_3d,
        u_1d,
        atol=1e-10,
        err_msg="Spatial Invariance failed: 3D reshape altered the unmixing matrix."
    )

    np.testing.assert_allclose(
        u_shuffled,
        u_1d,
        atol=1e-10,
        err_msg="Spatial Invariance failed: Shuffling pixels altered the unmixing matrix. The algorithm is order-dependent."
    )
