import numpy as np
from picasso.unmixing import compute_unmixing_matrix, mutual_information


def test_multichannel_blind_source_separation():
    """
    Testr 🔎: Verify the correctness of 3-channel blind source separation.

    This test validates that:
    1.  The unmixing algorithm can recover 3 independent sources from a mixed signal.
    2.  The unmixing matrix closely approximates the theoretical inverse of the mixing matrix (up to diagonal normalization).
    3.  The Total Mutual Information (sum of pairwise MI) is significantly reduced.
    4.  The recovered sources are highly correlated with the ground truth.

    Using Gamma distribution for sources to mimic fluorescence intensity (non-negative, skewed).
    Using a 3x3 mixing matrix with positive off-diagonals (bleed-through).
    """
    np.random.seed(1337)

    # 1. Generate Synthetic Data
    n_pixels = 50_000
    # Gamma distribution (k=2.0, theta=2.0)
    s1 = np.random.gamma(2.0, 2.0, n_pixels)
    s2 = np.random.gamma(2.0, 2.0, n_pixels)
    s3 = np.random.gamma(2.0, 2.0, n_pixels)

    sources = np.stack([s1, s2, s3])

    # Calculate Baseline Total Mutual Information (should be low)
    # Total Correlation approx sum of pairwise MI
    mi_baseline = (
        mutual_information(s1, s2)
        + mutual_information(s1, s3)
        + mutual_information(s2, s3)
    )

    # 2. Define Mixing Matrix M (Diagonally dominant, positive off-diagonals)
    # Simulates spectral bleed-through
    M = np.array([[1.0, 0.4, 0.2], [0.3, 1.0, 0.3], [0.2, 0.4, 1.0]])

    # 3. Mix Sources
    mixed_flat = M @ sources
    # Reshape to (C, N, 1) as expected by compute_unmixing_matrix for "image" input
    mixed = mixed_flat.reshape(3, n_pixels, 1)

    # Calculate Mixed Total MI
    m1, m2, m3 = mixed_flat[0], mixed_flat[1], mixed_flat[2]
    mi_mixed = (
        mutual_information(m1, m2)
        + mutual_information(m1, m3)
        + mutual_information(m2, m3)
    )

    # Verify mixing increased MI
    assert (
        mi_mixed > mi_baseline * 2
    ), f"Mixing failed to increase MI. Baseline: {mi_baseline:.4f}, Mixed: {mi_mixed:.4f}"

    # 4. Run Unmixing
    # Using a quantile of 0.0 to use all pixels (since we generated them cleanly)
    # But usually quantile is used to pick high signal pixels.
    # Since our Gamma sources have high signal naturally (and no background noise explicitly added other than inherent randomness),
    # using all pixels is fine.
    # However, to test the default behavior, let's use quantile=0.5 or 0.0.
    # Let's use 0.0 to be safe about sample size.
    u_computed = compute_unmixing_matrix(
        mixed,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=50,  # Should converge quickly
    )

    # 5. Derive Expected Unmixing Matrix
    # Theoretical inverse
    M_inv = np.linalg.inv(M)
    # Normalize diagonal to 1.0 to match picasso constraint
    # D * M_inv has 1s on diagonal => D_ii * (M_inv)_ii = 1 => D_ii = 1 / (M_inv)_ii
    D = np.diag(1.0 / np.diag(M_inv))
    u_expected = D @ M_inv

    # 6. Verify Unmixing Matrix
    # Check if computed matrix is close to expected
    # The algorithm uses approximation and finite samples, so allow some tolerance.
    # 0.1 is generous but ensures we are in the right ballpark.
    np.testing.assert_allclose(
        u_computed,
        u_expected,
        atol=0.1,
        err_msg=f"Unmixing matrix deviated from expectation.\nExpected:\n{u_expected}\nGot:\n{u_computed}",
    )

    # 7. Verify Total Correlation Reduction
    # Apply unmixing
    unmixed_flat = u_computed @ mixed_flat
    u1, u2, u3 = unmixed_flat[0], unmixed_flat[1], unmixed_flat[2]

    mi_unmixed = (
        mutual_information(u1, u2)
        + mutual_information(u1, u3)
        + mutual_information(u2, u3)
    )

    # Assert MI reduction
    # We expect MI to be close to baseline
    assert (
        mi_unmixed < mi_mixed * 0.2
    ), f"Unmixing failed to significantly reduce MI. Mixed: {mi_mixed:.4f}, Unmixed: {mi_unmixed:.4f}"

    assert (
        mi_unmixed <= mi_baseline + 0.1
    ), f"Unmixing did not recover independence. Baseline: {mi_baseline:.4f}, Unmixed: {mi_unmixed:.4f}"

    # 8. Verify Source Recovery (Correlation)
    # Compute correlation matrix between Original Sources and Recovered Sources
    # We expect diagonal to be high (close to 1.0)
    # Note: u_computed @ M is the effective transformation from Sources to Recovered.
    # T = U @ M. Since U ~ D @ M_inv, T ~ D.
    # So Recovered ~ D @ Sources.
    # Correlation is scale invariant, so corr(Recovered_i, Source_i) should be ~ 1.0.

    for i in range(3):
        corr = np.corrcoef(sources[i], unmixed_flat[i])[0, 1]
        assert (
            corr > 0.98
        ), f"Source {i} was not correctly recovered. Correlation: {corr:.4f}"
