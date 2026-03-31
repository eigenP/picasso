import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance (Bag of Pixels Guarantee) of the unmixing algorithm.

    This test validates that:
    The unmixing algorithm processes images strictly as a "bag of pixels", ignoring any
    spatial context or ordering. If the spatial positions of pixels are identically
    permuted (shuffled) across all channels, the resulting unmixing matrix must remain
    exactly the same.

    Why this matters:
    1. **Mathematical Property**: The algorithm relies on Mutual Information, which is
       computed from 1D and 2D intensity histograms. These histograms only depend on
       co-occurring intensity pairs, entirely invariant to their coordinate location.
    2. **Algorithmic Guarantee**: It proves the algorithm can natively support arbitrary
       spatial dimensions (1D spectra, 2D images, 3D volumes, temporal series) simply by
       flattening them, without requiring specialized spatial neighborhood logic.
    3. **Robustness**: Any accidental dependence on spatial gradients or local context
       would immediately break this invariant.
    """
    # Seed for reproducibility
    np.random.seed(42)

    # 1. Generate Synthetic Mixed Data (2D Images)
    # Using 100x100 pixels (10,000 pixels total)
    width, height = 100, 100
    n_pixels = width * height

    # Independent sources (Gamma distributed for positivity, common in imaging)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix M (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])

    # Mixed flattened signals
    mixed_flat = M @ sources

    # Reshape back to 2D image channels
    mixed_images = mixed_flat.reshape(2, height, width)

    # 2. Compute Baseline Unmixing Matrix
    # quantile=0.0 and max_samples=n_pixels ensures we test the exact algorithm
    # on the full mathematical dataset without any stochastic subsampling noise.
    u_base = compute_unmixing_matrix(
        [mixed_images[0], mixed_images[1]],
        verbose=False,
        max_iters=20, # Sufficient for convergence
        quantile=0.0,
        max_samples=n_pixels
    )

    # Ensure the baseline is non-trivial
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Spatially Shuffled Images
    # Generate a random permutation for the spatial indices
    spatial_permutation = np.random.permutation(n_pixels)

    # Flatten, permute identically for both channels, and reshape back
    shuffled_channel_0 = mixed_images[0].ravel()[spatial_permutation].reshape(height, width)
    shuffled_channel_1 = mixed_images[1].ravel()[spatial_permutation].reshape(height, width)

    shuffled_images = [shuffled_channel_0, shuffled_channel_1]

    # 4. Compute Unmixing Matrix on Shuffled Data
    u_shuffled = compute_unmixing_matrix(
        shuffled_images,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Spatial Invariance
    # The unmixing matrix should be identical because spatial order does not affect the histograms.
    # We use a strict tolerance since this is an exact property of the Mutual Information objective function.
    np.testing.assert_allclose(
        u_shuffled,
        u_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed: Shuffling pixel positions altered the unmixing matrix."
    )
