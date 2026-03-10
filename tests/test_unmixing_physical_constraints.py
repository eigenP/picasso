import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_constraints():
    """
    Testr 🔎: Verify the algorithm enforces strict physical positivity constraints.

    This test validates that the algorithm correctly models the physical reality
    of fluorescence imaging: crosstalk is strictly additive ($M_{ij} \ge 0$).
    Therefore, the unmixing matrix must have non-positive off-diagonal elements
    ($U_{ij} \le 0$) to subtract this crosstalk.

    If given strongly anti-correlated (adversarial) data, a naive mathematical
    optimization of Mutual Information would find a positive coefficient ($U_{ij} > 0$)
    to decouple the signals. However, this is physically impossible—one fluorophore
    cannot "absorb" photons from another channel to create negative crosstalk.

    The algorithm must recognize this unphysical attempt, clamp the coefficients
    to zero, and return the Identity matrix.

    This ensures the algorithm degrades gracefully to a null-op rather than
    hallucinating mathematically valid but scientifically impossible "negative
    fluorescence" artifacts.
    """
    np.random.seed(42)

    # 1. Setup: Generate Adversarial Data
    # We need strictly anti-correlated signals. If Ch1 goes up, Ch2 goes down.
    n_pixels = 50_000

    # Base signal
    base_signal = np.random.uniform(0.1, 0.9, n_pixels)

    # Ch1 is positively correlated with the base signal
    # Ch2 is negatively correlated (anti-correlated) with the base signal
    s1 = base_signal
    s2 = 1.0 - base_signal

    # Add noise to avoid pure degeneracy where MI estimation might break down
    # (e.g., if the scatter plot is a perfect 1D line, entropy binning can behave erratically)
    noise_level = 0.05
    s1 += np.random.normal(0, noise_level, n_pixels)
    s2 += np.random.normal(0, noise_level, n_pixels)

    # Ensure strictly positive values (simulating photon counts/intensities)
    s1 = np.clip(s1, 0.01, None)
    s2 = np.clip(s2, 0.01, None)

    # Reshape to (Channels, Pixels, 1) for the unmixing API
    adversarial_img = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Execution
    # Feed the anti-correlated data into the algorithm.
    U_computed = compute_unmixing_matrix(
        adversarial_img,
        quantile=0.0, # Use all pixels to avoid selection bias on anti-correlated tails
        max_samples=n_pixels,
        verbose=False
    )

    # 3. Verification
    # The algorithm should have attempted to find a positive coefficient,
    # hit the physical constraint clamp (mat_cumul[row, col] > 0.0 -> 0.0),
    # and returned the Identity matrix.

    # We use a strict tolerance because the constraint should be exact (0.0).
    np.testing.assert_allclose(
        U_computed,
        np.eye(2),
        atol=1e-10,
        err_msg=(
            "Physical Constraints failed: Algorithm hallucinated unphysical positive "
            "crosstalk when given anti-correlated data. It must return Identity."
        )
    )
