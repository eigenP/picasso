import numpy as np
from picasso.unmixing import minimize_mi

def test_independent_sources_rejected():
    """
    Test that minimize_mi rejects unmixing independent sources.
    Due to the empirical mutual information estimator noise floor, it shouldn't hallucinate.
    """
    np.random.seed(42)
    N = 10000
    # Create two independent sources
    x = np.random.gamma(2, 2, N)
    y = np.random.gamma(2, 2, N)

    alpha = minimize_mi(x, y, bins=32)
    # The statistical threshold should strictly bound out minor MI decreases
    # resulting from the empirical histogram estimator bias.
    assert alpha == 0.0

def test_dependent_sources_unmixed():
    """
    Test that minimize_mi accepts unmixing dependent/mixed sources
    where the reduction in mutual information exceeds the bias threshold.
    """
    np.random.seed(42)
    N = 10000
    x = np.random.gamma(2, 2, N)
    # y contains a true bleed-through of 0.5 * x
    y = x * 0.5 + np.random.gamma(2, 2, N)

    alpha = minimize_mi(x, y, bins=32)
    # It should find the unmixing coefficient and confidently clear the threshold
    assert alpha > 0.4
    assert alpha < 0.6
