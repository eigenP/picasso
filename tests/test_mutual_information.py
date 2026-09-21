import numpy as np
from picasso.unmixing import mutual_information

def test_mi_non_negative():
    """
    Verify that mutual information is mathematically non-negative (I(X;Y) >= 0).
    When 1D marginals are calculated independently of the 2D joint histogram,
    floating point edge effects can cause discrepancies that break the law
    of total probability, resulting in negative mutual information.
    """
    np.random.seed(42)

    # We test multiple iterations with data scales and offsets that are known
    # to hit floating-point edge clipping in histogram bounds.
    for i in range(100):
        x = np.random.uniform(0, 255, 1000)
        y = np.random.uniform(0, 100, 1000)

        # Test with a small number of bins to amplify discrete effects
        mi = mutual_information(x, y, bins=10)

        # We allow a very small tolerance for pure floating point summation differences
        assert mi >= -1e-12, f"Mutual Information was negative: {mi}"
