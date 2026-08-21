## 2024-05-24 - Empirical Mutual Information Optimization and Bias Correction

**Learning:** Optimizing empirical Mutual Information via a 1D line search on piecewise-constant 2D histograms (e.g., using algorithms like COBYLA) is fundamentally unprincipled. It often leads to converging on local noise and hallucinating false unmixing coefficients. Furthermore, any empirical Mutual Information estimate carries a statistical bias known as the Miller-Madow bias, which is approximately $K/(2N)$, where $K$ is the number of non-zero bins and $N$ is the number of samples. Optimizing empirical MI without bias correction results in overfitting the noise floor, causing the algorithm to 'unmix' independent variables unphysically.

**Action:** Replace unprincipled optimizers with Brent's method (`minimize_scalar(method='bounded')`) for robust 1D search, and explicitly apply the Miller-Madow bias threshold. If the theoretical mutual information reduction from the optimized coefficient does not exceed the statistical bias floor ($bins / (2N)$), the mathematical improvement is rejected, and the coefficient defaults to its initial value (`init_alpha`, generally 0.0). This ensures mathematically sound convergence and prevents hallucinating unphysical crosstalk.

## 2024-05-24 - The Law of Total Probability in Empirical Mutual Information

**Learning:** When computing empirical Mutual Information, the 1D marginal distributions must be derived directly from the joint 2D histogram (e.g., via summation over axes). Calculating 1D histograms independently can lead to mismatching subsets of data being binned (due to floating point edge effects and out-of-bounds clipping between the 1D and 2D estimators), which mathematically breaks the law of total probability and violates the non-negativity guarantee $I(X; Y) \ge 0$.

**Action:** Always derive marginal probabilities (1D histograms) directly from the joint probability matrix (2D histogram) via marginalization (`sum(axis=...)`) rather than using independent binning algorithms to ensure strict mathematical consistency.
