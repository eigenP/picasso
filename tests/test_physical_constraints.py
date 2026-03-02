import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_physical_non_positivity_constraint():
    """
    Testr 🔎: Verify the algorithmic intent of the Physical Positivity Constraint.

    💡 What: The algorithm explicitly constrains all off-diagonal unmixing coefficients
             to be non-positive ($U_{{ij}} \\le 0$). This test verifies that this constraint
             is strictly enforced even when an unconstrained optimization would prefer
             positive coefficients (e.g., when the inputs are anti-correlated).

    🎯 Why: In fluorescence microscopy, spectral crosstalk is strictly additive:
            channels can only *add* signal to each other ($M_{{ij}} \\ge 0$). Consequently,
            unmixing must always subtract signal. If the algorithm allowed $U_{{ij}} > 0$,
            it would imply negative fluorescence crosstalk, violating physical reality.
            Without this constraint, the algorithm might over-optimize by hallucinating
            unphysical "additive unmixing" to eliminate coincidental anti-correlations.

    🧪 How: We generate two independent synthetic sources and mix them with a *negative*
            mixing matrix (creating artificial anti-correlation). We ensure the resulting
            signals remain strictly positive (by adding a large background) to simulate valid
            imaging data. We then run the unmixing algorithm.

    📐 Theory: Pure MI minimization without constraints would attempt to remove this
               anti-correlation by finding a positive unmixing coefficient ($\\alpha > 0$).
               However, because the algorithm correctly models physical reality, it must
               clamp these coefficients to $0$, effectively refusing to "unmix" the
               physically impossible negative crosstalk, returning the Identity matrix.

    ⚠️ Catches: A bug where the bounds of the optimization (`step_mult * coef`) are not
                correctly enforced, allowing positive off-diagonal elements which would
                lead to unphysical signal amplification during unmixing.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Generate Independent Sources (Gamma distributed for positivity)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # 2. Apply Adversarial "Negative" Mixing
    # This creates strong anti-correlation.
    # A naive MI minimizer would want to add a fraction of Ch1 to Ch0 to cancel it.
    M_anti = np.array([
        [1.0, -0.6],
        [-0.4, 1.0]
    ])
    mixed_flat = M_anti @ sources

    # 3. Ensure signals are strictly positive and well above the background threshold (1e-6)
    # The minimum value might be negative due to the negative mixing.
    min_val = np.min(mixed_flat)
    if min_val < 0:
        # Add a massive, independent constant background to shift everything > 0
        # We add different constants to ensure we don't introduce trivial correlation
        mixed_flat[0, :] += abs(min_val) + 10.0
        mixed_flat[1, :] += abs(min_val) + 20.0

    # Reshape for API
    mixed_input = mixed_flat.reshape(2, n_pixels, 1)

    # 4. Compute Unmixing Matrix
    # Use quantile=0.0 and max_samples=n_pixels to use all data and avoid subsampling noise.
    U_computed = compute_unmixing_matrix(
        mixed_input,
        verbose=False,
        quantile=0.0,
        max_samples=n_pixels,
        max_iters=20
    )

    # 5. Verify the Constraint
    # We assert two things:
    # A. All off-diagonal elements MUST be exactly <= 0.0
    off_diagonals = U_computed[~np.eye(2, dtype=bool)]
    assert np.all(off_diagonals <= 0.0), \
        f"Physical constraint violated: Off-diagonal elements must be <= 0.0. Got:\n{U_computed}"

    # B. Because the input was pure anti-correlation, the optimal physical move is to do nothing.
    # Therefore, the unmixing matrix should be exactly Identity (or very close, clamped to 0).
    # We allow a small tolerance because MI estimation and coordinate descent
    # might produce tiny non-zero updates before clamping takes over completely
    # (or due to finite sampling). The key is that the off-diagonals don't grow large and negative
    # either (which would happen if it actually found a real correlation).
    np.testing.assert_allclose(
        U_computed,
        np.eye(2),
        atol=1e-3, # Allow small tolerance for numerical noise in the clamped optimization
        err_msg=f"Algorithm failed to clamp unphysical unmixing. Expected Identity, Got:\n{U_computed}"
    )
    print("\n  ✅ Physical Positivity Constraint Verified: Algorithm correctly refused to unmix anti-correlated signals.")
