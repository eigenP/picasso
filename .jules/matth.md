# MATTH'S JOURNAL — CRITICAL LEARNINGS ONLY

This journal captures deep lessons about applied mathematics, algorithmic correctness, and statistical modeling choices in this codebase.

## 2025-05-27 - [Thresholding Empirical Mutual Information via Asymptotic Bias]
**Learning:** Optimizing empirical Mutual Information (MI) calculated from finite discrete samples (like 2D histograms) is mathematically hazardous because finite-sample estimators of entropy are systematically biased. The raw uncorrected empirical MI will often show an apparent reduction in entropy just by fitting to the noise floor (aliasing artifacts), causing optimization algorithms to hallucinate unphysical unmixing coefficients for independent signals. However, statistical theory (specifically, the Miller-Madow correction) provides that the asymptotic bias is approximately `B / (2 * N)` where `B` is the number of non-zero bins. We use `bins / (2 * N)` as a conservative threshold for the observed reduction in the objective function.
**Action:** When performing 1D line search (e.g., Brent's method) over empirical mutual information, always compute the initial MI and the optimized MI, and strictly reject any updates where the reduction is less than or equal to the Miller-Madow bias threshold. This ensures the algorithm is statistically grounded and protects against hallucinating correlations that don't exist.
