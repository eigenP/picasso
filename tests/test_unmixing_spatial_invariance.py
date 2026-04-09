import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance of the unmixing algorithm.

    This test validates that:
    The unmixing algorithm is strictly independent of the spatial arrangement
    of pixels. Because it treats the image as a global "bag of pixels" and relies
    solely on the joint intensity distributions (Mutual Information), randomly
    permuting the pixels spatially must yield the exact same unmixing matrix.

    Why this matters:
    1. It proves the algorithm does not rely on local spatial correlations or gradients,
       which would be sensitive to image resolution or biological structure size.
    2. It guarantees that the function can correctly handle arbitrary input dimensions
       (1D spectra, 2D images, 3D volumes, or temporal stacks) without needing
       specialized spatial processing logic for each modality.
    """
    np.random.seed(42)

    # 1. Setup: Generate Synthetic Correlated Data in a 2D grid
    n_y, n_x = 100, 100
    n_pixels = n_y * n_x

    # Generate structured data (e.g., gradients) to ensure spatial arrangement exists
    y_grid, x_grid = np.mgrid[0:n_y, 0:n_x]

    # Source 1: mostly depends on Y + noise
    s1 = (y_grid / n_y) + np.random.gamma(2, 0.5, (n_y, n_x))
    # Source 2: mostly depends on X + noise
    s2 = (x_grid / n_x) + np.random.gamma(2, 0.5, (n_y, n_x))

    sources = np.stack([s1, s2])  # Shape: (2, 100, 100)

    # Mixing Matrix M
    M = np.array([[1.0, 0.4], [0.3, 1.0]])

    # Mix the sources (tensordot properly handles arbitrary spatial dims)
    mixed = np.tensordot(M, sources, axes=1)  # Shape: (2, 100, 100)

    # 2. Compute Baseline Unmixing Matrix
    # We use quantile=0.0 and max_samples=n_pixels to include all pixels deterministically,
    # avoiding random subsampling noise that might differ after permutation.
    u_base = compute_unmixing_matrix(
        mixed,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Ensure baseline is non-trivial
    assert not np.allclose(u_base, np.eye(2)), "Baseline unmixing matrix is trivial."

    # 3. Create Spatially Permuted Data
    # Flatten spatial dimensions, permute, then reshape back
    mixed_flat = mixed.reshape(2, -1)

    # Generate a single random permutation to apply to all channels identically.
    # This preserves the essential pixel-wise cross-channel correlations.
    perm = np.random.permutation(n_pixels)

    mixed_perm_flat = mixed_flat[:, perm]
    mixed_perm = mixed_perm_flat.reshape(2, n_y, n_x)

    # 4. Compute Unmixing Matrix on Permuted Data
    u_perm = compute_unmixing_matrix(
        mixed_perm,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Spatial Invariance
    # The matrices should be identical because the underlying distribution is identical.
    np.testing.assert_allclose(
        u_perm,
        u_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed. Permuting pixels spatially changed the unmixing matrix."
    )
