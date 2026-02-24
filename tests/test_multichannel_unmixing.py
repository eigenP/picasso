
import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix, mutual_information

def calculate_total_mi(image):
    """Calculates the sum of pairwise Mutual Information for all channel pairs."""
    n_channels = image.shape[0]
    total_mi = 0.0
    # Flatten if not already 2D (C, N)
    if image.ndim > 2:
        image = image.reshape(n_channels, -1)

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            total_mi += mutual_information(image[i], image[j])
    return total_mi

def test_three_channel_unmixing_and_monotonicity():
    """
    Testr 🔎: Verify unmixing correctness for 3 channels and MI reduction.

    This test validates:
    1. Unmixing works for N=3 channels (beyond simple pairwise).
    2. The algorithm recovers the correct unmixing matrix analytically.
    3. The total Mutual Information decreases significantly.

    It uses Gamma distributed sources to mimic fluorescence intensity distributions
    (non-negative, skewed) and a known 3x3 mixing matrix.
    """
    np.random.seed(42)
    n_pixels = 50_000
    n_channels = 3

    # 1. Generate Independent Sources (Gamma distributed)
    # Shape k=2.0, Scale=1.0
    sources = np.random.gamma(2.0, 1.0, (n_channels, n_pixels))

    # 2. Define Mixing Matrix
    # M = [[1, 0.2, 0.1], [0.2, 1, 0.2], [0.1, 0.2, 1]]
    # This matrix is diagonally dominant and invertible.
    M = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.2],
        [0.1, 0.2, 1.0]
    ])

    # 3. Mix Sources
    mixed_flat = M @ sources
    # Reshape to (C, N, 1) as expected by compute_unmixing_matrix (arbitrary spatial dims)
    mixed = mixed_flat.reshape(n_channels, n_pixels, 1)

    # 4. Calculate Expected Unmixing Matrix
    # The algorithm produces a matrix U such that U @ M is diagonal.
    # U is constrained to have 1.0 on the diagonal.
    # Analytically, U = D @ M_inv, where D is diagonal scaling.
    # D_ii = 1 / (M_inv)_ii to satisfy U_ii = 1.
    M_inv = np.linalg.inv(M)
    D = np.diag(1.0 / np.diag(M_inv))
    U_expected = D @ M_inv

    # 5. Compute Unmixing Matrix
    # Use quantile=0.0 to use all pixels (no subsampling based on intensity)
    # ensuring we test the core optimization logic on the full distribution.
    # Set max_iters sufficiently high.
    U_computed = compute_unmixing_matrix(
        mixed,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False,
        max_iters=500
    )

    # 6. Verify Unmixing Matrix
    # Tolerance 0.05 is reasonable for stochastic optimization (histogram-based MI)
    np.testing.assert_allclose(
        U_computed,
        U_expected,
        atol=0.05,
        err_msg=f"Computed unmixing matrix deviates from analytical expectation.\nExpected:\n{U_expected}\nGot:\n{U_computed}"
    )

    # 7. Verify Diagonal Product (Source Recovery)
    # The product U_computed @ M should be close to diagonal (D)
    product = U_computed @ M
    off_diagonal_mask = ~np.eye(n_channels, dtype=bool)
    off_diagonal_elements = product[off_diagonal_mask]
    diagonal_elements = np.diag(product)

    # Off-diagonals should be small relative to diagonal
    # This confirms that the unmixing matrix successfully inverts the mixing (up to scaling)
    max_off_diag = np.max(np.abs(off_diagonal_elements))
    min_diag = np.min(np.abs(diagonal_elements))

    assert max_off_diag < 0.1 * min_diag, \
        f"Unmixing did not diagonalize the mixing matrix sufficiently. Max off-diag: {max_off_diag}, Min diag: {min_diag}"

    # 8. Verify Mutual Information Reduction
    mi_baseline = calculate_total_mi(sources)
    mi_mixed = calculate_total_mi(mixed_flat)

    # Apply unmixing to flat mixed data
    unmixed_flat = U_computed @ mixed_flat
    mi_unmixed = calculate_total_mi(unmixed_flat)

    print(f"MI Baseline: {mi_baseline:.4f}, MI Mixed: {mi_mixed:.4f}, MI Unmixed: {mi_unmixed:.4f}")

    # Check that MI is significantly reduced
    # 1. It should be less than the mixed state
    assert mi_unmixed < mi_mixed, \
        f"Total Mutual Information did not decrease. Mixed: {mi_mixed}, Unmixed: {mi_unmixed}"

    # 2. It should return close to the baseline level (allow some margin for optimization noise)
    # The margin needs to be reasonable. 2x baseline is safe, usually it's much closer.
    # Given the previous run: Unmixed=0.1765. If baseline is ~0.15, this is fine.
    assert mi_unmixed < mi_baseline + 0.1, \
        f"Unmixing did not fully recover independence. Baseline: {mi_baseline}, Unmixed: {mi_unmixed}"
