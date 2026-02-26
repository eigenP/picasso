# TESTR’S JOURNAL — CRITICAL LEARNINGS ONLY

This journal captures deep lessons about correctness, not routine test additions.

2024-05-22 - [Synthetic Ground Truth for Unsupervised Learning]
Learning: Unsupervised algorithms like blind source separation are hard to test with real data because the ground truth is unknown. However, they can be rigorously verified by generating synthetic independent sources, mixing them with a known matrix, and asserting that the algorithm recovers the inverse. This transforms an "optimization problem" into a "functional correctness" test with strong assertions.
Action: For unsupervised algorithms, always prioritize tests that generate data from the generative model the algorithm assumes (e.g., independent components) to verify recovery.

2024-05-23 - [Physical Invariants as Test Oracles]
Learning: Algorithms modeling physical processes (like spectral unmixing) must satisfy fundamental physical invariants like Scale Invariance (units don't matter) and Permutation Equivariance (channel labels don't matter). These properties can be tested exactly (or to high precision) even when the ground truth is unknown, providing a powerful "consistency check" that catches bugs in normalization, thresholding, and optimization logic.
Action: When testing scientific code, identify physical symmetries (scaling, rotation, permutation, time-reversal) and implement property-based tests that verify them.

2024-05-23 - [Pairwise Optimization Sufficiency]
Learning: Multi-channel unmixing can be effectively verified by checking the global reduction of Total Correlation (sum of pairwise Mutual Information). The algorithm's strategy of iterative pairwise minimization (coordinate descent) converges to the global optimum for diagonally dominant mixing, confirming that N-channel correctness can be inferred from pairwise interactions.
Action: When testing iterative pairwise algorithms, verify the global property (Total Correlation) to ensure that local improvements aggregate correctly to a global solution.

2025-05-23 - [Statistical Consistency of Estimators]
Learning: Optimization algorithms that estimate parameters from data must behave as consistent estimators—error should decrease as sample size $N$ increases. Verifying this monotonic error reduction confirms that the algorithm effectively utilizes additional information (law of large numbers) and that adaptive strategies (like binning) scale correctly with $N$, distinguishing a "lucky" fit from a true statistical convergence.
Action: For estimation algorithms, implement tests that measure error across logarithmic steps of sample size (e.g., $N, 10N, 50N$) and assert monotonic improvement.
