import numpy as np

from picasso.unmixing import mutual_information, shannon_entropy
from fast_histogram import histogram1d, histogram2d

def test_mutual_information_positivity():
    """
    Verify that mutual information is always non-negative, even when floating-point
    edge cases cause independent 1D histograms to disagree with the 2D joint histogram.

    If 1D marginals are not correctly derived directly from the 2D joint histogram,
    floating point binning differences can violate the Law of Total Probability,
    leading to hallucinated negative mutual information.
    """
    # Create an edge case where clipping behavior might differ slightly
    x_test = np.array([0.0, 1.0, 0.999999999])
    y_test = np.array([0.0, 1.0, 1.0])

    # Check the updated mutual_information function directly
    mi = mutual_information(x_test, y_test, bins=2)
    assert mi >= 0.0, f"Mutual information was negative: {mi}"

    # Specifically demonstrate the floating point edge case failure
    # and prove our derivation fixes it.
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)

    c_xy = histogram2d(x_test, y_test, 2, (x_range, y_range))
    c_x_indep = histogram1d(x_test, 2, x_range)
    c_y_indep = histogram1d(y_test, 2, y_range)

    h_xy = shannon_entropy(c_xy)
    h_x_indep = shannon_entropy(c_x_indep)
    h_y_indep = shannon_entropy(c_y_indep)

    mi_indep = h_x_indep + h_y_indep - h_xy

    # We know in this specific case the independent calculation is incorrect
    # and might be negative or at least inconsistent.
    # The correct marginals MUST sum over the joint.
    c_x_marg = c_xy.sum(axis=1)
    c_y_marg = c_xy.sum(axis=0)

    h_x_marg = shannon_entropy(c_x_marg)
    h_y_marg = shannon_entropy(c_y_marg)

    mi_marg = h_x_marg + h_y_marg - h_xy

    assert mi_marg >= 0.0, "Derived marginal MI should be exactly non-negative"

    # With this edge case, they are actually functionally different due to clipping.
    # c_x_indep might count 0.999999999 differently than c_xy does for the x dimension.
    # We don't strictly assert mi_indep < 0 because it depends on the fast_histogram version,
    # but we assert mi_marg is always correct.
