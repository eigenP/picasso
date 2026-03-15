import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_unphysical_anti_correlation():
    """
    Testr 🔎: Verify that physical constraints correctly override mathematical optimization.

    This test validates that the algorithm refuses to unmix (returns Identity)
    when given explicitly adversarial, unphysical data: perfectly anti-correlated signals.

    Why this matters:
    1. **Physical Reality**: Fluorescence crosstalk is strictly additive. Channel 1 bleeding
       into Channel 2 means they will be positively correlated. "Negative crosstalk"
       does not exist in this physical system.
    2. **Mathematical Blindness**: The core optimization (minimizing Mutual Information)
       is blind to physics. If given anti-correlated data, an unconstrained MI minimizer
       would happily find a *positive* unmixing coefficient (which corresponds to negative
       crosstalk) because adding one channel to the other would "destroy" information
       and reduce MI.
    3. **Constraint Enforcement**: The algorithm contains a specific physical bound:
       `if mat_cumul[row, col] > 0.0: mat_cumul[row, col] = 0.0`
       This test verifies that this exact line of code successfully catches and rejects
       unphysical mathematical optima.

    If this test fails (i.e., the algorithm returns a matrix other than Identity),
    it means the algorithm is hallucinating unphysical behavior to satisfy an equation.
    """
    np.random.seed(42)

    # 1. Setup: Generate Adversarial Anti-Correlated Data
    n_pixels = 50_000

    # Create a base signal
    base_signal = np.random.uniform(1.0, 10.0, n_pixels)

    # Make Channel 1 high when base is high
    c1 = base_signal

    # Make Channel 2 low when base is high (anti-correlated)
    # Ensure it stays positive (valid physical intensities)
    c2 = 11.0 - base_signal

    # Add some independent noise to avoid perfectly degenerate bins which might
    # cause the MI estimator to fail for other reasons. We want the MI minimizer
    # to actually attempt to optimize.
    c1 = c1 + np.random.normal(0, 0.5, n_pixels)
    c2 = c2 + np.random.normal(0, 0.5, n_pixels)

    # Ensure strictly positive intensities
    c1 = np.clip(c1, 0.1, None)
    c2 = np.clip(c2, 0.1, None)

    # Shape: (Channels, Pixels, 1)
    anti_correlated_input = np.stack([c1, c2]).reshape(2, n_pixels, 1)

    # 2. Run Unmixing
    # Use quantile=0.0 and max_samples to use the whole dataset, avoiding stochastic noise
    U_computed = compute_unmixing_matrix(
        anti_correlated_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=20 # Should converge immediately to identity
    )

    # 3. Assert Physical Constraint Enforcement
    # The algorithm must reject this data and return the Identity matrix.
    # It must NOT return off-diagonal elements > 0.
    np.testing.assert_allclose(
        U_computed,
        np.eye(2),
        atol=1e-10, # Strict tolerance since it should explicitly clamp to 0.0
        err_msg=(
            "Physical bounds failed! Algorithm hallucinated unphysical "
            "negative crosstalk (positive unmixing coefficients) "
            "when given anti-correlated data."
        )
    )
