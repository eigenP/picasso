import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance (Dimensionality Agnosticism) of the unmixing algorithm.

    This test validates that:
    1. The unmixing algorithm does not rely on local spatial context or pixel adjacency.
    2. Randomly permuting all spatial coordinates (identically across channels) results
       in the exact same unmixing matrix.

    Why this matters:
    - It proves the core optimization operates purely on the joint intensity distribution
      (the "bag of pixels" assumption).
    - It mathematically guarantees that the algorithm supports arbitrary dimensionalities
      (1D spectra, 2D images, 3D volumes, or temporal stacks) without structural modifications,
      as any shape can be flattened into a 1D sequence without altering the joint statistics.
    """
    np.random.seed(42)
    n_pixels = 40_000

    # 1. Setup: Generate Synthetic Correlated Data
    # Use Gamma distributions to ensure positive, realistic intensity profiles
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mix sources to create crosstalk
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape into a 2D image
    side = int(np.sqrt(n_pixels))
    n_pixels_sq = side * side

    # Ensure exact square dimensions for the 2D test
    mixed_img = mixed_flat[:, :n_pixels_sq].reshape(2, side, side)

    # 2. Compute Baseline Unmixing Matrix
    U_base = compute_unmixing_matrix(
        list(mixed_img),
        verbose=False,
        max_iters=20, # Sufficient for convergence
        quantile=0.0, # Use all pixels to avoid stochastic selection noise from shifting percentiles
        max_samples=n_pixels_sq
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(U_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Spatially Scrambled Data
    # Generate a random permutation for all pixel indices
    perm_indices = np.random.permutation(n_pixels_sq)

    # Flatten the 2D image to apply the exact same permutation to both channels
    mixed_img_flat = mixed_img.reshape(2, -1)
    mixed_perm_flat = mixed_img_flat[:, perm_indices]

    # Reshape back to 2D
    mixed_perm_img = mixed_perm_flat.reshape(2, side, side)

    # 4. Compute Unmixing Matrix on Scrambled Data
    U_perm = compute_unmixing_matrix(
        list(mixed_perm_img),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels_sq
    )

    # 5. Assert Spatial Invariance
    # The unmixing matrix should be identical because Mutual Information is
    # computed on the unordered joint distribution of intensities.
    # We use a strict tolerance because this is a fundamental invariant of the algorithm's design.
    np.testing.assert_allclose(
        U_perm,
        U_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed. Scrambling pixel locations changed the unmixing matrix."
    )
