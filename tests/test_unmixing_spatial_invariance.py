import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_spatial_invariance():
    """
    Testr 🔎: Verify the Spatial Invariance of the unmixing algorithm.

    This test validates that the exact spatial arrangement of pixels does not influence
    the computed unmixing matrix. The unmixing algorithm operates on a "bag of pixels"
    by computing the mutual information of the global joint distribution of the channels.
    Therefore, randomly permuting the spatial dimensions identically across all channels
    should yield the exact same unmixing matrix.

    This is a critical invariant to verify because it proves:
    1. The optimization acts purely in the spectral domain.
    2. The algorithm makes no assumptions about the geometry or spatial continuity of the data.
    3. The `compute_unmixing_matrix` function safely supports inputs of arbitrary spatial
       dimensions (1D, 2D, 3D, temporal) simply by flattening them internally.

    Mathematically: $U(I) = U(P_{spatial}(I))$, where $P_{spatial}$ is any spatial permutation.
    """
    # 1. Setup: Generate Synthetic Correlated Data
    np.random.seed(42)

    # We use a 2D image shape to test spatial permutations
    shape = (100, 100)
    n_pixels = shape[0] * shape[1]

    # Independent sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape to (Channels, Y, X) for the API
    mixed = mixed_flat.reshape(2, *shape)

    # 2. Compute Baseline Unmixing Matrix
    # We set quantile=0.0 and max_samples=n_pixels so that ALL valid pixels are used.
    # If we sub-sample deterministically (via np.linspace on sorted or top % arrays),
    # shuffling the flattened array changes WHICH pixels are selected,
    # thereby perturbing the final matrix. By using all pixels, we bypass the subsampling logic
    # and strictly test the core unmixing optimization invariant.
    u_base = compute_unmixing_matrix(
        list(mixed),
        verbose=False,
        max_iters=20, # Sufficient for convergence
        quantile=0.0,
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Spatially Permuted Data
    # Generate a random permutation of the spatial indices
    perm_indices = np.random.permutation(n_pixels)

    # Apply the same spatial permutation to all channels
    mixed_permuted_flat = mixed_flat[:, perm_indices]
    mixed_permuted = mixed_permuted_flat.reshape(2, *shape)

    # 4. Compute Unmixing Matrix on Permuted Data
    u_permuted = compute_unmixing_matrix(
        list(mixed_permuted),
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Spatial Invariance
    # The unmixing matrix should be strictly identical.
    # We use a tight tolerance since this is an exact mathematical property of the "bag of pixels" approach.
    np.testing.assert_allclose(
        u_permuted,
        u_base,
        atol=1e-10,
        err_msg="Spatial Invariance failed. Randomly permuting spatial pixels changed the unmixing matrix."
    )
