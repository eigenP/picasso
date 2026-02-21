# TESTR’S JOURNAL — CRITICAL LEARNINGS ONLY

This journal captures deep lessons about correctness, not routine test additions.

2024-05-22 - [Synthetic Ground Truth for Unsupervised Learning]
Learning: Unsupervised algorithms like blind source separation are hard to test with real data because the ground truth is unknown. However, they can be rigorously verified by generating synthetic independent sources, mixing them with a known matrix, and asserting that the algorithm recovers the inverse. This transforms an "optimization problem" into a "functional correctness" test with strong assertions.
Action: For unsupervised algorithms, always prioritize tests that generate data from the generative model the algorithm assumes (e.g., independent components) to verify recovery.
