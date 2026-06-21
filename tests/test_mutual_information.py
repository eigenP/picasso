import numpy as np
from picasso.unmixing import mutual_information, shannon_entropy
from fast_histogram import histogram1d, histogram2d

def test_mutual_information_non_negative():
    """
    Test that the mutual information is strictly non-negative, even with
    pathological distributions that might cause floating point edge cases
    if 1D histograms were calculated independently.
    """
    # Create adversarial inputs that could trigger edge discrepancies
    # if marginals were calculated independently.
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)

    # Introduce identical max values to create spikes on bin boundaries
    x[:20] = x.max()
    y[:20] = y.max()

    mi = mutual_information(x, y, bins=10)

    # Mathematical invariant: I(X;Y) >= 0
    assert mi >= -1e-12, f"Mutual information was negative: {mi}"

def test_mutual_information_identical_signals():
    """
    Test that I(X;X) == H(X)
    """
    np.random.seed(42)
    x = np.random.uniform(0, 1, 1000)

    mi = mutual_information(x, x, bins=20)

    # Calculate H(X) directly
    x_range = (x.min(), x.max())
    c_x = histogram1d(x, bins=20, range=x_range)
    hx = shannon_entropy(c_x)

    np.testing.assert_allclose(mi, hx, rtol=1e-5, atol=1e-5)
