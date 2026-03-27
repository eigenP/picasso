import pytest
import numpy as np
from picasso.unmixing import (
    compute_unmixing_matrix,
    apply_unmixing_matrix,
    select_representative_pixels,
)

def test_select_representative_pixels_args():
    # Create two synthetic channels of small size
    c1 = np.random.rand(100, 100).astype(np.float32)
    c2 = np.random.rand(100, 100).astype(np.float32)
    images = [c1, c2]

    # Test max_samples as an absolute float count (e.g. 1e6) vs an int vs a fraction <= 1.0
    # The total number of pixels is 10,000.
    # 1. fraction <= 1.0
    res_frac = select_representative_pixels(images, quantile=0.0, max_samples=0.5, min_samples=100)
    # Expected: 0.5 * 10000 = 5000 samples
    assert res_frac.shape[1] == 5000, f"Expected 5000, got {res_frac.shape[1]}"

    # 2. absolute count as float (e.g. 1e6)
    # Total valid is 10000, max_samples is 1000000 -> we fall back and max out at 10000 valid pixels
    res_float_abs = select_representative_pixels(images, quantile=0.0, max_samples=1e6, min_samples=100)
    assert res_float_abs.shape[1] == 10000, f"Expected 10000, got {res_float_abs.shape[1]}"

    # 3. absolute count as int (e.g. 500)
    res_int = select_representative_pixels(images, quantile=0.0, max_samples=500, min_samples=100)
    # Since quantile is 0.0, all valid pixels are high-signal, so it subsamples exactly 500
    assert res_int.shape[1] == 500, f"Expected 500, got {res_int.shape[1]}"


def test_compute_unmixing_matrix_api():
    # Create synthetic mixed data
    # Source signals
    s1 = np.random.rand(50, 50).astype(np.float32)
    s2 = np.random.rand(50, 50).astype(np.float32)

    # Mixing matrix
    M = np.array([[1.0, 0.5], [0.3, 1.0]])
    stacked_s = np.stack([s1, s2])
    mixed = np.tensordot(M, stacked_s, axes=1)

    # Convert back to list of arrays
    mixed_list = [mixed[0], mixed[1]]

    # Test API passing list of arrays
    u_mat = compute_unmixing_matrix(
        mixed_list,
        max_iters=5,
        step_mult=0.1,
        verbose=False,
        return_iters=False,
        max_samples=1e6,  # Test the max_samples absolute float bug here too
        min_samples=100,
        quantile=0.5
    )
    assert u_mat.shape == (2, 2)
    assert np.allclose(np.diag(u_mat), [1.0, 1.0])

    # Test return_iters=True
    u_iters = compute_unmixing_matrix(
        mixed_list,
        max_iters=3,
        return_iters=True,
    )
    assert isinstance(u_iters, np.ndarray)
    assert u_iters.ndim == 3
    assert u_iters.shape[0] > 0

    # Test passing tuple
    mixed_tuple = (mixed[0], mixed[1])
    u_mat_tuple = compute_unmixing_matrix(mixed_tuple, max_iters=2)
    assert u_mat_tuple.shape == (2, 2)


def test_apply_unmixing_matrix_api():
    c1 = np.ones((10, 10))
    c2 = np.ones((10, 10)) * 2
    images = [c1, c2]

    U = np.array([[1.0, -0.5], [-0.5, 1.0]])

    unmixed = apply_unmixing_matrix(images, U)

    # Validate return type is list
    assert isinstance(unmixed, list)
    assert len(unmixed) == 2
    assert isinstance(unmixed[0], np.ndarray)

    # Validate correct values
    # unmixed[0] = 1*c1 + (-0.5)*c2 = 1 - 1 = 0
    assert np.allclose(unmixed[0], 0.0)
    # unmixed[1] = (-0.5)*c1 + 1*c2 = -0.5 + 2 = 1.5
    assert np.allclose(unmixed[1], 1.5)
