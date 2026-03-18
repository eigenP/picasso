import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_positivity_constraint():
    """
    Testr 🔎: Verify the Physical Positivity Constraint (Non-negative crosstalk).

    This test validates that the unmixing algorithm strictly enforces physical bounds,
    specifically that it refuses to hallucinate "negative crosstalk" (which would manifest
    as positive off-diagonal elements in the unmixing matrix).

    💡 What: When given strongly anti-correlated input signals, an unbounded Mutual
    Information optimizer might try to add them together (positive off-diagonals)
    to destroy their structure and minimize MI. The algorithm must clamp these updates.

    🎯 Why: In fluorescence microscopy, crosstalk is strictly additive (photons from dye A
    bleeding into channel B). It is physically impossible to have "negative" photons
    subtracted at the detector. Overriding pure mathematical optimization with physical
    priors prevents catastrophic unmixing artifacts on naturally anti-correlated
    biological structures (e.g., mutually exclusive compartments like nucleus vs cytoplasm).

    🧪 How: We generate adversarial synthetic data: two strongly anti-correlated signals.
    We pass this into the unmixing algorithm and verify that instead of minimizing MI
    via positive off-diagonals, it correctly recognizes the signals as either independent
    or physically invalid to unmix, clamping the matrix to Identity.

    📐 Theory: The constraint $U_{ij} \\le 0$ for $i \\neq j$ ensures that unmixing
    only ever *subtracts* scaled versions of other channels ($Y_{new} = Y - \\alpha X$,
    where $\\alpha \\ge 0$).

    ⚠️ Catches: Bugs where physical bounds are disabled, applied incorrectly (e.g.,
    wrong sign), or where mathematical optimization is allowed to override domain constraints.
    """
    # 1. Setup: Generate Adversarial Anti-correlated Data
    np.random.seed(42)
    n_pixels = 50_000

    # Base signal dictates spatial structure
    # A single shared baseline that varies
    base_structure = np.random.uniform(0.2, 0.8, n_pixels)

    # Signal 1 follows the base structure
    s1 = base_structure
    # Signal 2 is the exact inverse (strongly anti-correlated)
    s2 = 1.0 - base_structure

    # Add minor independent noise to prevent degenerate histogram binning artifacts
    # where all values collapse into single bins, which breaks MI estimation.
    s1 = s1 + np.random.normal(0, 0.05, n_pixels)
    s2 = s2 + np.random.normal(0, 0.05, n_pixels)

    # Ensure strictly positive values for physical realism
    s1 = np.clip(s1, 0.01, None)
    s2 = np.clip(s2, 0.01, None)

    # Combine into input image shape (Channels, Pixels, 1)
    adversarial_input = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # 2. Execution: Run Unmixing
    # We use all pixels (quantile=0.0) to avoid any selection artifacts,
    # and provide enough iterations for an unbounded optimizer to run amok.
    U = compute_unmixing_matrix(
        adversarial_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=10
    )

    # 3. Verification: Assert Physical Bounds
    # The expected behavior is that the algorithm tries to optimize, sees that it
    # would need positive off-diagonals, clamps them to 0, and thus returns the
    # Identity matrix (no unmixing applied).

    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-6,
        err_msg="Algorithm hallucinated unphysical negative crosstalk (positive off-diagonals) for anti-correlated inputs. The physical positivity constraint failed."
    )
