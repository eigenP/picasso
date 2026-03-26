import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance (Bag of Pixels) guarantee of the unmixing algorithm.

    This test validates that:
    The computed unmixing matrix is strictly invariant to the spatial arrangement
    of the pixels in the image.

    Why this matters:
    The algorithm claims to use a "bag of pixels" approach, meaning it relies entirely
    on the joint intensity distribution (Mutual Information) and not on spatial context
    (like edges, textures, or neighborhoods).
    By proving that randomly shuffling the pixels yields the exact same unmixing matrix,
    we guarantee that the algorithm is a pure point-wise statistical operation. This
    ensures it can safely and correctly process data of arbitrary dimensionality
    (1D spectra, 2D images, 3D volumes, or temporal stacks) without any subtle bugs
    introduced by spatial heuristics or coordinate-dependent processing.
    """
    np.random.seed(42)
    n_pixels = 20_000

    # 1. Setup: Generate Synthetic Mixed Data
    # Independent sources
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape into a 2D spatial image (Channels, Y, X)
    side_len = int(np.sqrt(n_pixels))
    mixed_image = mixed_flat[:, :side_len**2].reshape(2, side_len, side_len)
    actual_pixels = side_len**2

    # 2. Compute Baseline Unmixing Matrix
    # We use quantile=0.0 and max_samples to include all pixels, ensuring we test
    # the exact mathematical objective without stochastic subsampling noise.
    U_base = compute_unmixing_matrix(
        mixed_image,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=actual_pixels
    )

    assert not np.allclose(U_base, np.eye(2)), "Baseline is trivial."

    # 3. Apply Spatial Permutation (Shuffle)
    # We flatten the spatial dimensions, apply a random permutation, and reshape back.
    # The permutation must be identical for all channels to preserve pixel-wise correlations.
    perm_indices = np.random.permutation(actual_pixels)

    mixed_image_flat = mixed_image.reshape(2, actual_pixels)
    mixed_shuffled_flat = mixed_image_flat[:, perm_indices]
    mixed_shuffled = mixed_shuffled_flat.reshape(2, side_len, side_len)

    # 4. Compute Unmixing Matrix on Shuffled Image
    U_shuffled = compute_unmixing_matrix(
        mixed_shuffled,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=actual_pixels
    )

    # 5. Assert Exact Spatial Invariance
    # The optimization relies purely on histograms of intensities, which are invariant
    # to the order of the samples. Therefore, the unmixing matrix must be identical.
    np.testing.assert_allclose(
        U_shuffled,
        U_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed. Shuffling pixels changed the unmixing matrix."
    )
