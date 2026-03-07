import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_positivity_constraint():
    """
    Testr 🔎: Verify the physical positivity constraint (additive crosstalk).

    💡 What: The algorithm must enforce that off-diagonal elements of the unmixing
       matrix are strictly non-positive (U_{ij} <= 0). When given anti-correlated
       signals (where a naive statistical model would suggest U_{ij} > 0 to minimize MI),
       the algorithm must clamp the update and return the Identity matrix.

    🎯 Why: Fluorescence crosstalk is strictly additive (light leakage adds signal,
       never subtracts it). Therefore, unmixing must strictly subtract (U_{ij} <= 0).
       Allowing positive off-diagonals implies unphysical "negative crosstalk".

    🧪 How: We generate two strongly anti-correlated signals. A pure statistical
       decorrelation would try to add them to minimize mutual information. We verify
       that the algorithm correctly refuses to do this, clamping the unmixing matrix
       to Identity.

    📐 Theory: Unmixing physically corresponds to U * X, where U_{ii} = 1 and U_{ij} <= 0.
       Anti-correlated inputs should hit this constraint immediately.

    ⚠️ Catches: Bugs in the constraint logic (e.g., removing the `> 0` clamp or
       clamping in the wrong direction), which would lead to hallucinated signals
       in anti-correlated datasets.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Generate Anti-Correlated Signals
    # We create a base signal, and make the two channels opposing.
    # We add independent noise so the signals aren't perfectly degenerate,
    # avoiding mutual information binning artifacts.
    base = np.random.uniform(0, 10, n_pixels)

    # Channel 1 is proportional to base + noise
    s1 = base + np.random.normal(0, 1, n_pixels)

    # Channel 2 is inversely proportional to base + noise
    s2 = (10 - base) + np.random.normal(0, 1, n_pixels)

    # Ensure all values are strictly positive to avoid thresholding issues
    s1 = np.clip(s1, 0.1, None)
    s2 = np.clip(s2, 0.1, None)

    sources = np.stack([s1, s2])

    # Reshape to (Channels, Pixels, 1) for the unmixing API
    mixed_img = sources.reshape(2, n_pixels, 1)

    # 2. Run Unmixing
    # We use quantile=0.0 and max_samples=n_pixels to use all data and avoid subsampling noise.
    # A naive optimizer would find a positive coefficient because adding s1 and s2 yields a
    # more uniform distribution, changing MI.
    U_computed = compute_unmixing_matrix(
        mixed_img,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=10
    )

    # 3. Verify Positivity Constraint / Clamping
    # Because the signals are anti-correlated, the optimal mathematical unmixing (without constraints)
    # would have U_{12} > 0 and U_{21} > 0.
    # The physical constraint must clamp these to 0, resulting in the Identity matrix.
    np.testing.assert_allclose(
        U_computed,
        np.eye(2),
        atol=1e-10,
        err_msg="Positivity constraint failed: Algorithm hallucinated unphysical negative crosstalk for anti-correlated inputs."
    )
