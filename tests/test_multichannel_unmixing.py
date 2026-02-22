import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix, mutual_information

def calculate_total_pairwise_mi(data):
    """
    Calculate the sum of pairwise Mutual Information between all channels.
    data: (n_channels, n_pixels)
    """
    n_channels = data.shape[0]
    total_mi = 0.0
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Use 100 bins for consistency with typical unmixing defaults
            mi = mutual_information(data[i], data[j], bins=100)
            total_mi += mi
    return total_mi

def test_three_channel_unmixing_and_information_reduction():
    """
    Testr 🔎: Verify multichannel unmixing and information reduction.

    This test validates that:
    1.  The algorithm scales to 3 channels (verifying the iterative pairwise approach).
    2.  Total pairwise Mutual Information is significantly reduced (information theoretic guarantee).
    3.  The computed unmixing matrix matches the analytical inverse of the mixing matrix (functional correctness).

    Uses Beta(2, 5) distributions to simulate bounded, skewed fluorescence signals.
    """
    np.random.seed(42)

    # 1. Setup: 3 Independent Sources
    n_pixels = 20_000
    # Beta(2, 5) is skewed towards 0, bounded [0, 1]
    s1 = np.random.beta(2, 5, n_pixels)
    s2 = np.random.beta(2, 5, n_pixels)
    s3 = np.random.beta(2, 5, n_pixels)
    sources = np.stack([s1, s2, s3]) # (3, N)

    # Baseline MI
    baseline_mi = calculate_total_pairwise_mi(sources)

    # 2. Mixing
    # Mixing matrix M (Identity with stronger bleed-through to ensure significant MI increase)
    # M = [[1.0, 0.4, 0.2],
    #      [0.4, 1.0, 0.4],
    #      [0.2, 0.4, 1.0]]
    M = np.array([
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.4],
        [0.2, 0.4, 1.0]
    ])
    mixed = M @ sources # (3, N)

    # Mixed MI
    mixed_mi = calculate_total_pairwise_mi(mixed)

    # Assert Mixing increased MI
    # With stronger mixing, we expect a clear increase
    assert mixed_mi > baseline_mi * 2.0, \
        f"Mixing should significantly increase Mutual Information. Baseline: {baseline_mi}, Mixed: {mixed_mi}"

    # 3. Execution
    # Reshape for function input: (C, Y, X) -> (C, N, 1)
    mixed_input = mixed.reshape(3, n_pixels, 1)

    # Use quantile=0.0 to include all pixels, avoiding subsampling artifacts in this test
    # max_samples >= n_pixels ensures no subsampling
    computed_U = compute_unmixing_matrix(
        mixed_input,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=True
    )

    # 4. Verification

    # A. Analytical Correctness
    # Expected U is related to M_inv.
    # U @ M should be diagonal.
    # M_inv = inv(M).
    # U should have 1s on diagonal.
    # U_expected = D @ M_inv where D makes diagonal 1.
    M_inv = np.linalg.inv(M)
    D = np.diag(1.0 / np.diag(M_inv))
    expected_U = D @ M_inv

    # Check structural match
    # Tolerance 0.1 for approximation error in optimization
    np.testing.assert_allclose(
        computed_U,
        expected_U,
        atol=0.1,
        err_msg=f"Computed unmixing matrix deviates from analytical inverse.\nExpected:\n{expected_U}\nGot:\n{computed_U}"
    )

    # B. Information Reduction
    # Apply computed unmixing
    # Ensure unmixing matrix shape matches mixed data
    unmixed_flat = computed_U @ mixed
    unmixed_mi = calculate_total_pairwise_mi(unmixed_flat)

    # Assert significant reduction
    # We expect MI to drop significantly
    assert unmixed_mi < mixed_mi * 0.5, \
        f"Unmixing did not sufficiently reduce MI. Mixed: {mixed_mi:.4f}, Unmixed: {unmixed_mi:.4f}"

    # Assert close to baseline (allow some margin for imperfect optimization and binning noise)
    # Margin 0.3 (0.1 per pair approx)
    assert unmixed_mi <= baseline_mi + 0.3, \
        f"Unmixing did not fully recover independence. Baseline: {baseline_mi:.4f}, Result: {unmixed_mi:.4f}"
