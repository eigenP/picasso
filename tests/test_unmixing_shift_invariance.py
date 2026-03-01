import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_shift_invariance():
    """
    Testr 🔎: Verify the Shift Invariance (Background Invariance) of the unmixing algorithm.

    This test validates that adding a constant background to the channels does not change
    the resulting unmixing matrix. In fluorescence microscopy, non-zero baselines
    (camera offset, ambient light, autofluorescence) are extremely common.

    Mathematically, the algorithm relies on minimizing Mutual Information.
    Mutual Information is translation-invariant: $I(X; Y) = I(X + c_1; Y + c_2)$.
    The unmixing operation subtracts scaled versions of one channel from another:
    $Y_{new} = Y - \alpha X$.
    If we add backgrounds $c_X$ and $c_Y$, then:
    $Y_{new} = (Y + c_Y) - \alpha(X + c_X) = (Y - \alpha X) + (c_Y - \alpha c_X)$.
    This results in a shifted version of the unmixed signal, but its Mutual Information
    with $X + c_X$ remains identical to the unshifted case.

    Thus, the optimal $\alpha$ (which defines the unmixing matrix) must be exactly
    the same regardless of constant additive backgrounds, provided the signals
    are not thresholded out or saturated.

    This property ensures the algorithm is robust to poor or variable background
    subtraction preprocessing.
    """
    # 1. Setup: Generate Synthetic Correlated Data
    np.random.seed(42)
    n_pixels = 10_000

    # Independent sources (Gamma distributed for positivity)
    # Adding a small constant to avoid zeros which might get thresholded by the 1e-6 background check
    s1 = np.random.gamma(2, 2, n_pixels) + 0.1
    s2 = np.random.gamma(2, 2, n_pixels) + 0.1
    sources = np.stack([s1, s2])

    # Mixing Matrix (Introduction of crosstalk)
    M = np.array([[1.0, 0.4], [0.2, 1.0]])
    mixed_flat = M @ sources

    # Reshape to (Channels, Pixels, 1) for the API
    mixed = mixed_flat.reshape(2, n_pixels, 1)

    # 2. Compute Baseline Unmixing Matrix
    u_base = compute_unmixing_matrix(
        mixed,
        verbose=False,
        max_iters=20, # Sufficient for convergence
        quantile=0.0, # Use all pixels to avoid stochastic selection noise from shifting percentiles
        max_samples=n_pixels
    )

    # Verify baseline is non-trivial (off-diagonals are non-zero)
    assert not np.allclose(u_base, np.eye(2)), \
        "Baseline unmixing matrix is trivial (Identity). Use more correlated data."

    # 3. Create Shifted Data
    # Add an independent, massive constant background to each channel
    c1 = 50.0
    c2 = 100.0

    # Ensure the shape matches for broadcasting
    shifts = np.array([c1, c2]).reshape(2, 1, 1)
    mixed_shifted = mixed + shifts

    # 4. Compute Unmixing Matrix on Shifted Data
    u_shifted = compute_unmixing_matrix(
        mixed_shifted,
        verbose=False,
        max_iters=20,
        quantile=0.0,
        max_samples=n_pixels
    )

    # 5. Assert Shift Invariance
    # The unmixing matrix should be identical because MI is translation invariant.
    # We use a very strict tolerance since this is an exact mathematical property of the objective function.
    np.testing.assert_allclose(
        u_shifted,
        u_base,
        atol=1e-10,
        err_msg="Shift Invariance failed. Adding a constant background changed the unmixing matrix."
    )
