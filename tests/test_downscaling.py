import numpy as np
from picasso.unmixing import _downscale_local_mean, compute_unmixing_matrix

def test_downscale_local_mean():
    # Create a 4x4 array
    image = np.array([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14],
        [11, 12, 15, 16]
    ], dtype=float)

    # Test downscaling by factor of 2 in both dimensions
    factors = (2, 2)
    downscaled = _downscale_local_mean(image, factors)

    expected = np.array([
        [(1+2+3+4)/4, (5+6+7+8)/4],
        [(9+10+11+12)/4, (13+14+15+16)/4]
    ])

    np.testing.assert_array_equal(downscaled, expected)

def test_downscale_local_mean_truncation():
    # Test that uneven shapes are properly truncated
    image = np.array([
        [1, 2, 5],
        [3, 4, 7],
        [9, 10, 13]
    ], dtype=float)

    factors = (2, 2)
    downscaled = _downscale_local_mean(image, factors)

    expected = np.array([
        [(1+2+3+4)/4]
    ])

    np.testing.assert_array_equal(downscaled, expected)

def test_compute_unmixing_matrix_downscale_kwarg():
    # Simple synthetic test for the kwarg
    np.random.seed(42)
    img1 = np.random.rand(10, 10)
    img2 = np.random.rand(10, 10)
    images = [img1, img2]

    # If the kwarg doesn't crash and returns a 2x2 matrix, the pass-through works
    matrix = compute_unmixing_matrix(
        images,
        downscale=(2, 2),
        max_iters=2,
    )

    assert matrix.shape == (2, 2)
    assert not np.isnan(matrix).any()

def test_compute_unmixing_matrix_downscale_int_kwarg():
    np.random.seed(42)
    img1 = np.random.rand(10, 10, 10)
    img2 = np.random.rand(10, 10, 10)
    images = [img1, img2]

    matrix = compute_unmixing_matrix(
        images,
        downscale=2,  # should apply as (2, 2, 2)
        max_iters=2,
    )

    assert matrix.shape == (2, 2)
