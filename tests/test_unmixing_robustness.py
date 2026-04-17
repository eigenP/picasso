
import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix, select_representative_pixels, minimize_mi

def test_unmixing_small_samples():
    """Test unmixing robustness with very small number of samples."""
    np.random.seed(42)
    # create 2 channels, 100 pixels total
    n_pixels = 100
    s1 = np.random.rand(n_pixels)
    s2 = np.random.rand(n_pixels)
    image = np.stack([s1, s2]) # (2, 100)

    # Run unmixing with default min_samples=1000
    # The robustness fix should use all 100 pixels (since 100 < 1000)
    # And adaptive binning should use small number of bins.

    matrix = compute_unmixing_matrix(
        list(image), verbose=True, max_iters=5)

    assert matrix.shape == (2, 2)
    # Check diagonals are 1
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    # Check off-diagonals are <= 0
    assert np.all(matrix[~np.eye(2, dtype=bool)] <= 0.0)

def test_select_representative_pixels_fallback():
    """Test that select_representative_pixels falls back to valid pixels if high-signal count is low."""
    np.random.seed(42)

    # Create image with 1000 pixels.
    # 2 channels
    image = np.zeros((2, 1000)) + 0.1 # Background (valid)
    image[:, :10] = 10.0 # High signal (10 pixels)

    # Quantile 0.99 -> Top 1% = 10 pixels.
    # So n_high_signal should be around 10.
    # We request min_samples=50.
    # The function should return 50 pixels (fallback to valid ones).

    pixels = select_representative_pixels(
        list(image),
        quantile=0.99,
        min_samples=50,
        max_samples=50,
        verbose=True
    )

    # We expect 50 pixels returned
    assert pixels.shape[1] == 50

    # Check that the 10 high signal pixels are included
    # (Since they are max intensity, they should be in the top 50 fallback)
    # The max value in 'pixels' should be 10.0
    assert np.max(pixels) == 10.0
    # The min value should be 0.1
    assert np.min(pixels) == 0.1

def test_minimize_mi_avoids_noise_fitting():
    """
    Testr 🔎: Verify robustness of minimize_mi to finite-sample noise.

    This test validates that when given purely independent noise sources,
    the optimization does not hallucinate a positive unmixing coefficient.
    The new implementation uses Brent's method with a threshold based on
    the Miller-Madow bias `(bins - 1)^2 / (2 * N)` to prevent "fitting the noise floor"
    of the empirical mutual information estimator.
    """
    np.random.seed(42)

    # Use enough pixels to make the bias calculation reliable
    N = 10_000
    x = np.random.gamma(2, 2, N)
    y = np.random.gamma(2, 2, N)

    # With purely independent sources, we expect the unmixing coefficient to be exactly 0.0
    alpha = minimize_mi(x, y, bins=100)

    assert alpha == 0.0, f"Algorithm hallucinated unmixing coefficient {alpha} on independent noise."

def test_minimize_mi_detects_real_signal():
    """
    Testr 🔎: Verify that true crosstalk is still detected properly.
    """
    np.random.seed(42)

    N = 10_000
    x = np.random.gamma(2, 2, N)
    # y contains 10% crosstalk from x
    y = np.random.gamma(2, 2, N) + 0.1 * x

    alpha = minimize_mi(x, y, bins=100)

    # We expect a coefficient near 0.1, it shouldn't be clamped to 0
    assert alpha > 0.05, f"Algorithm failed to detect real crosstalk, returned {alpha}."
