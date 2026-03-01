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
            # Using standard 100 bins
            mi = mutual_information(data[i], data[j], bins=100)
            total_mi += mi
    return total_mi

def test_unmixing_monotonic_convergence():
    """
    Testr 🔎: Verify the algorithmic monotonicity of the unmixing optimization.

    This test validates that:
    The core algorithm (coordinate descent on Mutual Information) monotonically
    decreases the objective function (Total Correlation) at every single iteration.

    Why this matters:
    1. **Optimization Correctness**: If the Total Correlation ever significantly increases,
       it implies the pairwise `minimize_mi` step is broken or that local updates
       are interfering detrimentally with the global objective.
    2. **Algorithmic Stability**: Guaranteed descent ensures that the algorithm
       is stable and will eventually reach a minimum, preventing infinite oscillations.

    We use `return_iters=True` to inspect the unmixing matrix at each iteration,
    apply it to the mixed data, and calculate the Total Correlation. We expect
    $MI_{t} \\le MI_{t-1}$ for all $t$.
    """
    np.random.seed(42)

    # 1. Setup: Generate Synthetic Correlated Data (3 channels to test global reduction)
    n_pixels = 50_000

    # Independent sources (Gamma distributed)
    s1 = np.random.gamma(2, 2, n_pixels)
    s2 = np.random.gamma(2, 2, n_pixels)
    s3 = np.random.gamma(2, 2, n_pixels)
    sources = np.stack([s1, s2, s3])

    # Mix sources to create crosstalk
    # Using a 3x3 matrix to ensure we test multi-channel global behavior
    M = np.array([
        [1.0, 0.4, 0.2],
        [0.3, 1.0, 0.5],
        [0.2, 0.3, 1.0]
    ])
    mixed_flat = M @ sources

    # Reshape for API (Channels, Pixels, 1)
    mixed_input = mixed_flat.reshape(3, n_pixels, 1)

    # Calculate initial baseline Total Correlation
    initial_mi = calculate_total_pairwise_mi(mixed_flat)

    # 2. Run Unmixing and collect iteration history
    # max_iters=15 is enough to observe a trajectory of reductions
    # quantile=0.0 and max_samples=n_pixels ensures we test on the exact same data
    # that we calculate MI on, avoiding stochastic subsampling noise.
    mats = compute_unmixing_matrix(
        mixed_input,
        verbose=False,
        max_iters=15,
        quantile=0.0,
        max_samples=n_pixels,
        return_iters=True
    )

    # Ensure we actually have iterations to test
    assert len(mats) > 1, "Algorithm converged too quickly to test monotonicity."

    # 3. Verify Monotonicity
    print(f"\nTestr 🔎: Tracking Total Correlation (MI) over {len(mats)} iterations...")
    print(f"  Initial MI: {initial_mi:.6f}")

    prev_mi = initial_mi

    for t, mat in enumerate(mats):
        # Apply the intermediate unmixing matrix
        unmixed_flat = mat @ mixed_flat

        # Calculate current Total Correlation
        current_mi = calculate_total_pairwise_mi(unmixed_flat)
        print(f"  Iter {t+1:2d} MI: {current_mi:.6f}")

        # The optimization must be monotonic.
        # However, because of finite binning, interpolation, and minor numerical noise
        # during coordinate descent, we allow a tiny tolerance for "increases" that are
        # effectively flat.
        tolerance = 1e-4

        # Check monotonic decrease: current_mi <= prev_mi + tolerance
        assert current_mi <= prev_mi + tolerance, \
            f"Monotonicity violated at iteration {t+1}. MI increased from {prev_mi:.6f} to {current_mi:.6f}"

        prev_mi = current_mi

    # Finally, verify that the overall reduction is significant
    assert prev_mi < initial_mi * 0.5, \
        f"Algorithm failed to significantly reduce Total Correlation. Initial: {initial_mi:.6f}, Final: {prev_mi:.6f}"
    print("  ✅ Monotonicity Verified.")
