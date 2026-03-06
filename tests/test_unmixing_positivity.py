import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_physical_positivity_constraint():
    """
    Testr 🔎: Verify the Physical Positivity Constraint of the unmixing algorithm.

    This test validates that:
    The algorithm enforces a strict physical positivity constraint ($U_{ij} \\le 0$ for $i \\neq j$).
    Because fluorescence crosstalk is strictly additive (photons from channel A bleed into channel B),
    the true mixing matrix $M$ must be non-negative, and thus the corresponding unmixing operations
    must only *subtract* signal (negative off-diagonal elements in $U$).

    If the algorithm is fed *anti-correlated* signals (e.g., fluorophores that are mutually exclusive
    in space), a naive correlation-based algorithm might try to *add* signal (positive off-diagonals)
    to enforce independence. The Picasso unmixing algorithm must refuse to do this, recognizing that
    such relationships are physically impossible for crosstalk, and clamp the coefficients to 0
    (returning the Identity matrix).

    Why this matters:
    Without this constraint, the algorithm could hallucinate unphysical negative crosstalk,
    destroying real biological mutually exclusive patterns (like mutually exclusive cell types).
    """
    np.random.seed(42)
    n_pixels = 50_000

    # 1. Setup: Generate Strongly Anti-Correlated Data
    # Base signal determines presence of the structures
    base = np.random.uniform(20, 80, n_pixels)

    # Source 1 is present when base is high
    s1 = base + np.random.normal(0, 2, n_pixels)

    # Source 2 is present when base is low (mutually exclusive / anti-correlated)
    s2 = 100 - base + np.random.normal(0, 2, n_pixels)

    sources = np.stack([s1, s2])

    # Verify the setup is actually anti-correlated
    corr = np.corrcoef(sources[0], sources[1])[0, 1]
    assert corr < -0.9, "Setup failed: sources are not strongly anti-correlated."

    mixed_input = sources.reshape(2, n_pixels, 1)

    # 2. Compute Unmixing Matrix
    # We expect the algorithm to attempt to find an alpha to minimize MI.
    # An unconstrained optimization might find a positive alpha.
    # The constrained algorithm should clamp it to 0.
    u_computed = compute_unmixing_matrix(
        mixed_input,
        quantile=0.0, # Use all pixels
        max_samples=n_pixels,
        verbose=False
    )

    # 3. Assert Positivity Constraint
    # The algorithm must return the Identity matrix (no unmixing applied)
    # because it cannot resolve the anti-correlation with physical (subtractive) operations.
    np.testing.assert_allclose(
        u_computed,
        np.eye(2),
        atol=1e-5,
        err_msg="Algorithm violated physical positivity constraint! It hallucinated negative crosstalk for anti-correlated inputs."
    )
