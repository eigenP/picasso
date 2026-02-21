
import numpy as np
from picasso.unmixing import compute_unmixing_matrix, mutual_information


def test_blind_source_separation_synthetic():
    """
    Testr 🔎: Verify the fundamental correctness of the blind source separation algorithm.

    This test validates that:
    1. Independent sources are preserved (unmixing matrix ~ Identity).
    2. Mixed sources are correctly unmixed to recover the original independent components.
    3. Mutual Information is minimized as claimed by the algorithm.

    Using synthetic data (Uniform noise) ensures strict independence and avoids
    incidental correlations found in natural images.
    """
    # Seed for reproducibility
    np.random.seed(42)

    # 1. Generate Synthetic Independent Sources
    # 50k pixels provides enough statistical power for stable MI estimation
    n_pixels = 50_000
    s1 = np.random.uniform(0, 1, n_pixels)
    s2 = np.random.uniform(0, 1, n_pixels)

    # Shape: (Channels, Pixels) -> Reshaped to (C, N, 1) for the function
    sources = np.stack([s1, s2]).reshape(2, n_pixels, 1)

    # Calculate Baseline Mutual Information
    # Ideally close to 0, but finite sampling creates a floor ~0.14 with 100 bins
    mi_baseline = mutual_information(sources[0], sources[1])

    # 2. Verify Identity Case (Zero-Mixing Response)
    # If sources are already independent, the algorithm should return Identity
    u_identity = compute_unmixing_matrix(
        sources,
        quantile=0.0,  # Use all pixels to test optimization logic purely
        max_samples=n_pixels,
        verbose=False
    )

    # Allow very small tolerance for numerical noise in optimization
    np.testing.assert_allclose(
        u_identity,
        np.eye(2),
        atol=1e-4,
        err_msg="Algorithm failed to return Identity for already independent sources."
    )

    # 3. Create Mixed Signal
    # Mixing Matrix M
    # Resulting signal: O = M * S
    # Unmixing target: U * O = S (up to scaling)
    # Theoretical U should have off-diagonals equal to -M_off if diagonals are normalized to 1
    M = np.array([[1.0, 0.5], [0.3, 1.0]])

    # Apply mixing
    # Need to reshape for matrix multiplication (2, N)
    sources_flat = sources.reshape(2, -1)
    mixed_flat = M @ sources_flat
    mixed = mixed_flat.reshape(2, n_pixels, 1)

    # Calculate MI of Mixed Signal
    mi_mixed = mutual_information(mixed[0], mixed[1])

    # Verify Mixing increased MI
    assert mi_mixed > mi_baseline * 2, \
        f"Mixing did not sufficiently increase Mutual Information. Baseline: {mi_baseline}, Mixed: {mi_mixed}"

    # 4. Perform Unmixing
    u_computed = compute_unmixing_matrix(
        mixed,
        quantile=0.0,
        max_samples=n_pixels,
        verbose=False
    )

    # 5. Verify Unmixing Matrix
    # Expected off-diagonals are -0.5 and -0.3
    expected_u = np.array([[1.0, -0.5], [-0.3, 1.0]])

    # Tolerance of 0.1 allows for stochastic approximation errors while catching gross failures
    np.testing.assert_allclose(
        u_computed,
        expected_u,
        atol=0.1,
        err_msg=f"Unmixing matrix deviated from expectation.\nExpected:\n{expected_u}\nGot:\n{u_computed}"
    )

    # 6. Verify Mutual Information Reduction
    # Apply computed unmixing to mixed signal
    unmixed_flat = u_computed @ mixed_flat

    mi_unmixed = mutual_information(unmixed_flat[0], unmixed_flat[1])

    # Check that MI returned to baseline levels
    # We allow a small margin (0.05) above baseline due to imperfect optimization
    assert mi_unmixed < mi_mixed, \
        f"Unmixing failed to reduce Mutual Information. Mixed: {mi_mixed}, Unmixed: {mi_unmixed}"

    assert mi_unmixed <= mi_baseline + 0.05, \
        f"Unmixing did not fully recover independence. Baseline: {mi_baseline}, Result: {mi_unmixed}"
