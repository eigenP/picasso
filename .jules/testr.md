# TESTR’S JOURNAL — CRITICAL LEARNINGS ONLY

This journal captures deep lessons about correctness, not routine test additions.

2024-05-22 - [Synthetic Ground Truth for Unsupervised Learning]
Learning: Unsupervised algorithms like blind source separation are hard to test with real data because the ground truth is unknown. However, they can be rigorously verified by generating synthetic independent sources, mixing them with a known matrix, and asserting that the algorithm recovers the inverse. This transforms an "optimization problem" into a "functional correctness" test with strong assertions.
Action: For unsupervised algorithms, always prioritize tests that generate data from the generative model the algorithm assumes (e.g., independent components) to verify recovery.

2024-05-24 - [Multichannel Correctness and Pairwise Minimization]
Learning: Pairwise mutual information minimization is sufficient to unmix N>2 channels if the algorithm iteratively updates the global unmixing matrix. Verifying this requires checking the "Total Mutual Information" (sum of pairwise MIs) reduction and comparing the unmixing matrix against the theoretical inverse of the mixing matrix (normalized). This confirms that local pairwise optimization leads to global source separation.
Action: For N-channel separation problems, test with N>2 to catch potential conflicts in pairwise updates and verify global convergence metrics like Total Correlation.
