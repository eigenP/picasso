import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_rejects_unphysical_negative_crosstalk():
    """
    Testr 🔎: Verify the physical positivity constraint (rejection of unphysical negative crosstalk).

    💡 What: The algorithm must enforce that off-diagonal unmixing coefficients are non-positive ($U_{ij} \\le 0$).
    🎯 Why: In fluorescence microscopy, spectral crosstalk is strictly additive ($M_{ij} \\ge 0$).
             Therefore, the unmixing matrix can only *subtract* signal from other channels
             to correct for it ($U_{ij} \\le 0$).
             If the algorithm is fed inputs where *adding* channels together (which implies
             negative crosstalk, $M_{ij} < 0$) would mathematically reduce Mutual Information,
             the optimization must hit its physical bounds and clamp the update to 0.
    🧪 How: We construct a system with *negative* crosstalk (anti-mixing). We mix
             two independent sources with a negative off-diagonal matrix. The purely
             mathematical inverse would have positive off-diagonals. We assert that
             the algorithm correctly refuses this unphysical solution, clamps the
             coefficients, and returns the Identity matrix.
    📐 Theory: $U = M^{-1}$. If $M = [[1, -0.5], [-0.5, 1]]$, then $U$ would have positive off-diagonals.
             The physical constraint $U_{ij} \\le 0$ overrides this.
    ⚠️ Catches: Bugs in the constraint-clamping logic (`mat_cumul[row, col] = 0.0`)
             inside the simultaneous update loop.
    """
    np.random.seed(42)
    n_pixels = 50_000

    # Independent sources
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2])

    # Unphysical "anti-mixing" matrix with negative crosstalk
    # M_ij < 0 is impossible in fluorescence, but tests the bounds logic.
    M_unphysical = np.array([
        [1.0, -0.5],
        [-0.5, 1.0]
    ])

    # Mix sources
    mixed_flat = M_unphysical @ sources

    # Ensure signals are still positive to avoid thresholding issues
    mixed_flat = np.clip(mixed_flat, a_min=0.1, a_max=None)

    adversarial_input = mixed_flat.reshape(2, n_pixels, 1)

    # Compute unmixing matrix
    U = compute_unmixing_matrix(
        adversarial_input,
        verbose=False,
        max_iters=10,
        quantile=0.0,
        max_samples=n_pixels
    )

    # Mathematically, MI would be minimized by U with positive off-diagonals.
    # The algorithm should clamp these to 0, returning the Identity matrix.
    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-8,
        err_msg="Algorithm hallucinated unphysical crosstalk. It failed to clamp positive U coefficients."
    )

    # Explicitly verify the constraint logic holds
    assert U[0, 1] <= 0.0, "Physical constraint violated: U[0, 1] > 0"
    assert U[1, 0] <= 0.0, "Physical constraint violated: U[1, 0] > 0"
