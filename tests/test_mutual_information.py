import numpy as np
from picasso.unmixing import mutual_information

def test_mutual_information_non_negative():
    """
    Test that mutual information is strictly non-negative, even in adversarial
    cases with points perfectly on bin boundaries that previously caused
    independent 1D histograms to disagree with the 2D joint histogram.
    """
    np.random.seed(42)

    # Create adversarial inputs
    x = np.random.uniform(0, 10, 100)
    y = np.random.uniform(0, 10, 100)

    # Add points perfectly on bin boundaries to trigger floating point binning artifacts
    bin_edges = np.linspace(x.min(), x.max(), 101)
    x = np.concatenate([x, bin_edges])
    y = np.concatenate([y, bin_edges])

    np.random.shuffle(x)
    np.random.shuffle(y)

    # Check that MI is non-negative
    mi = mutual_information(x, y, bins=100)
    assert mi >= -1e-12, f"Expected MI to be >= 0, but got {mi}"
