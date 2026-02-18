import numpy as np
import pytest
import skimage.data
from skimage.util import img_as_float
from picasso.unmixing import compute_unmixing_matrix
from skimage.transform import downscale_local_mean

def test_cells3d_unmixing():
    try:
        data = skimage.data.cells3d()
        z = data.shape[0] // 2
        initial = img_as_float(data[z, ...])
    except Exception as e:
        print(f"Failed to load cells3d: {e}")
        # Fallback to synthetic data
        from skimage.data import binary_blobs
        # Create 2 channels
        c1 = binary_blobs(length=256, blob_size_fraction=0.1, n_dim=2, seed=1)
        c2 = binary_blobs(length=256, blob_size_fraction=0.1, n_dim=2, seed=2)
        initial = np.stack([c1, c2]).astype(float)

    # make up a mixing matrix
    mixing_matrix = np.array([[1.0, 0.6], [0.7, 1.0]])
    mixed = np.einsum("ij,jkl->ikl", mixing_matrix, initial)

    # downscale as in the example to speed up
    mixed_downscaled = downscale_local_mean(mixed, (1, 2, 2))

    mat_iters = compute_unmixing_matrix(
        mixed_downscaled, verbose=True, return_iters=True, max_iters=100
    )
    unmixing_matrix = mat_iters[-1]

    print("Computed unmixing matrix:")
    print(unmixing_matrix)

    # Expected unmixing matrix for [[1.0, 0.6], [0.7, 1.0]] is roughly [[1.0, -0.6], [-0.7, 1.0]]
    # The example notebook result was [-0.70022008  1.        ]]

    expected_matrix = np.array([[1.0, -0.6], [-0.7, 1.0]])

    # Check if the result is close enough
    # Note: Optimization might not be perfect, so use a reasonable tolerance.
    # The diagonal elements are forced to 1.0, so check off-diagonal.

    np.testing.assert_allclose(unmixing_matrix, expected_matrix, atol=0.1)
