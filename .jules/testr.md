# TESTR’S JOURNAL — CRITICAL LEARNINGS ONLY

This journal captures deep lessons about correctness, not routine test additions.

2024-05-22 - [Synthetic Ground Truth for Unsupervised Learning]
Learning: Unsupervised algorithms like blind source separation are hard to test with real data because the ground truth is unknown. However, they can be rigorously verified by generating synthetic independent sources, mixing them with a known matrix, and asserting that the algorithm recovers the inverse. This transforms an "optimization problem" into a "functional correctness" test with strong assertions.
Action: For unsupervised algorithms, always prioritize tests that generate data from the generative model the algorithm assumes (e.g., independent components) to verify recovery.

2024-05-24 - [Multichannel Mutual Information Floor]
Learning: When unmixing >2 channels, the "zero mutual information" ideal is unattainable in finite samples (baseline MI ~0.15-0.2 for 50k samples). A strict monotonicity check (MI_unmixed < MI_mixed) combined with a proximity check to the baseline (MI_unmixed < Baseline + epsilon) is the correct way to verify convergence, rather than expecting MI ~ 0.
Action: Always measure the baseline MI of the generated independent sources to set a realistic target for the unmixing algorithm, rather than using an arbitrary small constant.
