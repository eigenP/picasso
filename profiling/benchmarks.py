import marimo

__generated_with = "0.9.33"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import time
    import numpy as np
    from scipy.optimize import minimize_scalar, fmin_cobyla
    import matplotlib.pyplot as plt
    from skimage import data
    from picasso.unmixing import mutual_information
    return data, fmin_cobyla, minimize_scalar, mo, mutual_information, np, plt, time


@app.cell
def __(data, np):
    try:
        # Load 2-channel 3D image: (60, 2, 256, 256) -> (Z, C, Y, X)
        cells_3d = data.cells3d()
    except Exception:
        # Fallback to random if cells3d is not available (e.g. download failed)
        np.random.seed(42)
        cells_3d = np.random.rand(60, 2, 256, 256)

    image_flat = cells_3d.reshape(cells_3d.shape[1], -1).astype(float)

    # Normalize to [0, 1]
    image_flat[0] /= image_flat[0].max()
    image_flat[1] /= image_flat[1].max()

    true_alpha = 3.75

    x_source = image_flat[0]
    y_pure = image_flat[1]
    y_mixed = y_pure + (true_alpha * x_source)

    bins = 100
    upper_bound = 10.0
    return bins, cells_3d, image_flat, true_alpha, upper_bound, x_source, y_mixed, y_pure


@app.cell
def __(fmin_cobyla, minimize_scalar, mutual_information, np):
    def run_benchmark(x, y, bins, upper_bound, method):
        evals = 0
        alphas = []
        errors = []
        costs = []

        def func(alpha):
            nonlocal evals
            evals += 1
            # COBYLA might pass an array instead of a scalar
            if isinstance(alpha, np.ndarray):
                a_val = float(alpha.item())
            else:
                a_val = float(alpha)
            cost = mutual_information(x, y - a_val * x, bins=bins)
            alphas.append(a_val)
            costs.append(cost)
            return cost

        if method == 'brent':
            res = minimize_scalar(
                func,
                bounds=(0.0, upper_bound),
                method='bounded',
                options={'xatol': 1e-4}
            )
            return float(res.x), evals, alphas, costs

        elif method == 'cobyla':
            cons = [lambda a: a, lambda a: upper_bound - a]
            res = fmin_cobyla(
                func=func,
                x0=np.array([0.0]),
                cons=cons,
                rhobeg=1.0/bins,
                rhoend=1e-8,
                disp=0
            )
            # COBYLA returns an ndarray; extract item correctly
            if isinstance(res, np.ndarray):
                res_val = res.item()
            else:
                res_val = res
            return float(res_val), evals, alphas, costs
    return run_benchmark,


@app.cell
def __(
    bins,
    run_benchmark,
    time,
    true_alpha,
    upper_bound,
    x_source,
    y_mixed,
):
    t0_brent = time.perf_counter()
    alpha_brent, nfev_brent, alphas_brent, costs_brent = run_benchmark(
        x_source, y_mixed, bins, upper_bound, 'brent'
    )
    t_brent = time.perf_counter() - t0_brent

    t0_cobyla = time.perf_counter()
    alpha_cobyla, nfev_cobyla, alphas_cobyla, costs_cobyla = run_benchmark(
        x_source, y_mixed, bins, upper_bound, 'cobyla'
    )
    t_cobyla = time.perf_counter() - t0_cobyla

    err_brent = abs(true_alpha - alpha_brent)
    err_cobyla = abs(true_alpha - alpha_cobyla)
    return (
        alpha_brent,
        alpha_cobyla,
        alphas_brent,
        alphas_cobyla,
        costs_brent,
        costs_cobyla,
        err_brent,
        err_cobyla,
        nfev_brent,
        nfev_cobyla,
        t0_brent,
        t0_cobyla,
        t_brent,
        t_cobyla,
    )


@app.cell
def __(
    alpha_brent,
    alpha_cobyla,
    err_brent,
    err_cobyla,
    mo,
    nfev_brent,
    nfev_cobyla,
    t_brent,
    t_cobyla,
    true_alpha,
):
    mo.md(
        f"""
        ### Optimization Benchmark Results

        **True Alpha:** `{true_alpha}`

        #### Brent's Method (Bounded)
        *   **Estimated Alpha:** `{alpha_brent:.5f}`
        *   **Absolute Error:** `{err_brent:.5f}`
        *   **Function Evaluations:** `{nfev_brent}`
        *   **Time Taken:** `{t_brent:.4f}s`

        #### COBYLA
        *   **Estimated Alpha:** `{alpha_cobyla:.5f}`
        *   **Absolute Error:** `{err_cobyla:.5f}`
        *   **Function Evaluations:** `{nfev_cobyla}`
        *   **Time Taken:** `{t_cobyla:.4f}s`
        """
    )
    return


@app.cell
def __(alphas_brent, alphas_cobyla, plt, true_alpha):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _ax1.plot(alphas_brent, label="Brent", marker='o', markersize=4)
    _ax1.plot(alphas_cobyla, label="COBYLA", marker='x', markersize=4)
    _ax1.axhline(true_alpha, color='r', linestyle='--', label="True Alpha")
    _ax1.set_xlabel("Function Evaluation Step")
    _ax1.set_ylabel("Alpha Estimate")
    _ax1.set_title("Optimization Trajectory (Alpha)")
    _ax1.legend()

    # Calculate errors over time
    _brent_err = [abs(a - true_alpha) for a in alphas_brent]
    _cobyla_err = [abs(a - true_alpha) for a in alphas_cobyla]

    _ax2.plot(_brent_err, label="Brent", marker='o', markersize=4)
    _ax2.plot(_cobyla_err, label="COBYLA", marker='x', markersize=4)
    _ax2.set_xlabel("Function Evaluation Step")
    _ax2.set_ylabel("Absolute Error")
    _ax2.set_title("Optimization Trajectory (Error)")
    _ax2.set_yscale('log')
    _ax2.legend()

    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


if __name__ == "__main__":
    app.run()
