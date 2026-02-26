import numpy as np
import pytest
from picasso.unmixing import compute_unmixing_matrix

def test_unmixing_consistency_with_sample_size():
    """
    Testr 🔎: Verify the Statistical Consistency of the unmixing estimator.

    This test validates that the algorithm behaves as a consistent statistical estimator:
    as the number of samples $N$ increases, the estimation error of the unmixing matrix
    should decrease. This confirms that the algorithm effectively utilizes additional
    information to converge towards the true analytical solution.

    We use a Monte Carlo approach with multiple trials per sample size to smooth out
    stochastic variance in the data generation and optimization process.
    """
    # 1. Setup: Define Ground Truth
    # Mixing Matrix M (Introduction of crosstalk)
    # Using a symmetric mixing for simplicity, but asymmetric works too.
    # M = [[1.0, 0.6],
    #      [0.4, 1.0]]
    M = np.array([[1.0, 0.6], [0.4, 1.0]])

    # Analytical Unmixing Matrix U_true
    # The unmixing matrix U must satisfy U @ M = Diagonal.
    # With U having 1s on the diagonal:
    # U = [[1, a], [b, 1]]
    # (1, a) @ (0.6, 1)^T = 0.6 + a = 0 => a = -0.6
    # (b, 1) @ (1, 0.4)^T = b + 0.4 = 0 => b = -0.4
    expected_U = np.array([[1.0, -0.6], [-0.4, 1.0]])

    # 2. Define Sample Sizes to Test
    # We expect error to decrease as we go from 2k -> 10k -> 50k samples.
    sample_sizes = [2000, 10000, 50000]
    n_trials = 5

    # Store Mean Absolute Errors (MAE) for each sample size
    maes = []

    np.random.seed(42) # Ensure reproducibility

    print("\nTestr 🔎: Verifying Statistical Consistency...")

    for n_samples in sample_sizes:
        trial_errors = []
        for _ in range(n_trials):
            # Generate Independent Sources (Gamma distributed)
            # Gamma is good because it's positive and skewed, like fluorescence
            s1 = np.random.gamma(2, 2, n_samples)
            s2 = np.random.gamma(2, 2, n_samples)
            sources = np.stack([s1, s2])

            # Mix sources
            mixed = M @ sources

            # Reshape for API (Channels, Pixels, 1)
            # We treat it as 1D image of length N
            mixed_input = mixed.reshape(2, n_samples, 1)

            # Compute Unmixing Matrix
            # quantile=0.0: Use ALL pixels. We want to test the estimator's
            # behavior on the full dataset N, not a subsample.
            # max_samples=n_samples: Ensure no subsampling occurs.
            computed_U = compute_unmixing_matrix(
                mixed_input,
                quantile=0.0,
                max_samples=n_samples,
                verbose=False,
                max_iters=50 # Sufficient for convergence
            )

            # Calculate Error (Mean Absolute Error on off-diagonals)
            # We focus on off-diagonals since diagonals are fixed to 1.0
            error = np.mean(np.abs(computed_U - expected_U))
            trial_errors.append(error)

        # Average error for this sample size
        avg_mae = np.mean(trial_errors)
        maes.append(avg_mae)
        print(f"  N={n_samples:5d} -> MAE={avg_mae:.6f}")

    # 3. Verify Monotonic Decrease
    # We expect error to decrease as N increases.
    # Check that each MAE is smaller than the previous one.
    # We allow a small tolerance or strict inequality?
    # Given the gap between 2k, 10k, 50k, strict inequality should hold
    # if the estimator is consistent and trials are sufficient.

    # Check N=2000 vs N=10000
    assert maes[1] < maes[0], \
        f"Error did not decrease from N={sample_sizes[0]} ({maes[0]:.6f}) to N={sample_sizes[1]} ({maes[1]:.6f})"

    # Check N=10000 vs N=50000
    assert maes[2] < maes[1], \
        f"Error did not decrease from N={sample_sizes[1]} ({maes[1]:.6f}) to N={sample_sizes[2]} ({maes[2]:.6f})"

    # 4. Verify Convergence
    # The final error should be reasonably small.
    # For N=50k, error should be quite low (e.g., < 0.05).
    assert maes[-1] < 0.05, \
        f"Estimator did not converge sufficiently. Final MAE ({maes[-1]:.6f}) is too high."

    print("  ✅ Consistency Verified: Error decreases monotonically with sample size.")
