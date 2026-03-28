import math
import sys
from typing import Union, Sequence, Literal

import numpy as np
import numpy.typing as npt
from fast_histogram import histogram2d, histogram1d
from scipy.optimize import fmin_cobyla, minimize_scalar
from skimage.util import img_as_float
from tqdm import tqdm


def select_representative_pixels(
    images: Sequence[npt.NDArray],
    quantile: float = 0.95,
    max_samples: Union[float, int] = 100_000,
    min_samples: int = 1_000,
    verbose: bool = False,
) -> npt.NDArray:
    n_channels = len(images)
    n_pixels = images[0].size

    try:
        image = np.stack(images, axis=0)
    except MemoryError:
        raise MemoryError("Not enough memory to stack the input images. Please provide smaller images or fewer channels.")

    # Flatten the image
    image_flat = image.reshape(n_channels, -1)

    # 1. Saturation Check
    # Identify saturated pixels if the input is integer type
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        # Pixel is saturated if ANY channel is saturated
        saturated_mask = np.any(image_flat == max_val, axis=0)
    else:
        saturated_mask = np.zeros(image_flat.shape[1], dtype=bool)

    # Filter out saturated pixels
    valid_indices = np.where(~saturated_mask)[0]

    if len(valid_indices) == 0:
        if verbose:
            print("Warning: All pixels are saturated.", file=sys.stdout)
        # Fallback: return everything converted to float
        return img_as_float(image_flat)

    # 2. Convert valid pixels to float
    # We only convert the non-saturated pixels to save memory/time
    valid_pixels_raw = image_flat[:, valid_indices]
    valid_pixels = img_as_float(valid_pixels_raw)

    # 3. Background Check
    # Filter out very low intensity (background) pixels based on L2 norm
    intensities = np.linalg.norm(valid_pixels, axis=0)
    # Threshold 1e-6 is heuristic for "near zero"
    background_mask = intensities < 1e-6

    # Keep non-background pixels
    keep_mask = ~background_mask
    valid_pixels = valid_pixels[:, keep_mask]

    if valid_pixels.shape[1] == 0:
        if verbose:
            print(
                "Warning: All non-saturated pixels are background.",
                file=sys.stdout,
            )
        # Fallback: return the saturated-filtered pixels (even if background)
        # or just return empty? Returning empty breaks things downstream.
        # Let's return the non-saturated set.
        return img_as_float(valid_pixels_raw)

    # 4. Top Percentile Selection
    # Select pixels that are in the top percentile for *any* channel
    n_valid = valid_pixels.shape[1]
    selected_mask = np.zeros(n_valid, dtype=bool)

    for c in range(n_channels):
        channel_data = valid_pixels[c, :]
        # Calculate threshold for this channel
        threshold = np.percentile(channel_data, quantile * 100)
        selected_mask |= channel_data >= threshold

    high_signal_pixels = valid_pixels[:, selected_mask]
    n_high_signal = high_signal_pixels.shape[1]

    # 5. Subsampling
    # Determine target sample count
    if isinstance(max_samples, float):
        if max_samples <= 1.0:
            # Interpret as a fraction of the ORIGINAL total pixels
            target_samples = int(max_samples * n_pixels)
        else:
            # Float provided as absolute count (e.g. 1e6)
            target_samples = int(max_samples)
    else:
        target_samples = max_samples

    # Ensure minimum samples
    target_samples = max(target_samples, min_samples)

    # 6. Fallback if not enough high signal pixels
    if n_high_signal < min_samples and n_valid > n_high_signal:
        if verbose:
            print(
                f"Warning: Only {n_high_signal} high-signal pixels found (threshold={quantile}). "
                f"Falling back to top {min(n_valid, target_samples)} valid pixels.",
                file=sys.stdout,
            )
        # Select top valid pixels based on max channel intensity
        # We want to fill up to target_samples (or at least min_samples?)
        # Let's target `target_samples` since we are falling back.

        needed = target_samples
        if needed > n_valid:
            needed = n_valid

        # Sort by max intensity
        max_intensities = np.max(valid_pixels, axis=0)
        # We want largest indices
        # usage of argpartition is faster than argsort for top k
        indices = np.argpartition(max_intensities, -needed)[-needed:]
        final_pixels = valid_pixels[:, indices]

    elif n_high_signal > target_samples:
        indices = np.linspace(0, n_high_signal - 1, target_samples, dtype=int)
        final_pixels = high_signal_pixels[:, indices]
        if verbose:
            print(
                f"Subsampled {n_high_signal} high-signal pixels to {target_samples} pixels.",
                file=sys.stdout,
            )
    else:
        final_pixels = high_signal_pixels
        if verbose:
            print(
                f"Using all {n_high_signal} high-signal pixels.",
                file=sys.stdout,
            )

    return final_pixels



def _downscale_local_mean(image: npt.NDArray, factors: tuple[int, ...]) -> npt.NDArray:
    """
    Fast block reduction downscaling using pure NumPy.
    Truncates the remainder if dimensions are not divisible by factors.
    """
    if image.ndim != len(factors):
        raise ValueError(
            f"Downscale factors length ({len(factors)}) must match image dimensions ({image.ndim})."
        )

    # 1. Calculate the new shape, truncating the remainder
    new_shape = tuple(s // f for s, f in zip(image.shape, factors))

    # 2. Crop the image to exactly fit the blocks
    slices = tuple(slice(0, n * f) for n, f in zip(new_shape, factors))
    cropped = image[slices]

    # 3. Reshape to interleave block dimensions
    reshaped_dims = []
    for n, f in zip(new_shape, factors):
        reshaped_dims.extend([n, f])

    reshaped = cropped.reshape(reshaped_dims)

    # 4. Average over the factor axes (which end up at odd indices: 1, 3, 5...)
    axes_to_mean = tuple(range(1, 2 * image.ndim, 2))

    return reshaped.mean(axis=axes_to_mean)


def shannon_entropy(a: npt.NDArray) -> float:
    """
    This runs very often, so we should do our best to make this fast.
    See 'profiling/shannon-entropy.ipynb' for why this was chosen
    """
    # seems like flattened arrays are faster, but .flatten() produces a copy so we
    # stick with .ravel()
    a = a.ravel()
    a /= a.sum()
    a = a[a > 0]
    a *= np.log2(a)
    return -a.sum().item()


def compute_optimal_bins(n_samples: int) -> int:
    """
    Computes the optimal number of bins for 2D histogram estimation of mutual information.
    Target ~10 samples per joint bin: B^2 = N / 10 => B = sqrt(N / 10).
    Constrains the bins to an array of "clean" divisors of 256 to prevent aliasing artifacts
    with 8-bit image data.
    """
    target_bins = int(math.sqrt(n_samples / 10))
    allowed_bins = np.array([8, 16, 32, 64, 128, 256])

    valid_options = allowed_bins[allowed_bins <= target_bins]
    if len(valid_options) == 0:
        return 8
    return int(valid_options[-1])


def mutual_information(x: npt.NDArray, y: npt.NDArray, *, bins=100) -> float:
    x = x.ravel()
    y = y.ravel()

    x_range = (x.min(), x.max())
    y_range = (y.min(), y.max())

    # TODO: even though fast-histogram is pretty dang fast, consider boost-histogram?
    #  measure perf diff
    c_xy = histogram2d(x, y, bins, (x_range, y_range))
    c_x = histogram1d(x, bins, x_range)
    c_y = histogram1d(y, bins, y_range)

    h_xy = shannon_entropy(c_xy)
    h_x = shannon_entropy(c_x)
    h_y = shannon_entropy(c_y)

    return h_x + h_y - h_xy


def regional_mi(x: npt.NDArray, y: npt.NDArray) -> float:
    """
    FIXME not optimized, but doesn't give us better results than regular mi, so not
     worth it at the moment
    """
    x = np.copy(x)
    y = np.copy(y)

    x /= x.max()
    y /= y.max()

    r = 1

    stack = []
    for ri in range(2 * r):
        for rj in range(2 * r):
            stack.append(x[ri : -2 * r + ri, rj : -2 * r + rj])
    for ri in range(2 * r):
        for rj in range(2 * r):
            stack.append(y[ri : -2 * r + ri, rj : -2 * r + rj])
    stack = np.stack(stack)
    stack = np.reshape(stack, (stack.shape[0], -1))

    dim, n_points = stack.shape[:2]
    hdim = dim // 2

    mean = np.mean(stack, axis=1, keepdims=True)
    stack -= mean
    cov = stack @ stack.T / n_points
    h_xy = math.log(np.linalg.det(cov))
    h_x = math.log(np.linalg.det(cov[:hdim, :hdim]))
    h_y = math.log(np.linalg.det(cov[hdim:, hdim:]))
    return h_x + h_y - h_xy


def minimize_mi(
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    init_alpha: float = 0.0,
    bins: int = 100,
    upper_bound: Union[float, None] = None,
    method: Literal['brent', 'cobyla'] = 'brent',
) -> float:
    """
    Minimizes the mutual information between x and (y - alpha * x) to find
    the optimal unmixing coefficient alpha.
    """
    def func(alpha: float) -> float:
        # Evaluate the cost function (Mutual Information)
        return mutual_information(x, y - alpha * x, bins=bins)

    if method == 'brent':
        # Brent's method: Golden Section Search + Parabolic Interpolation.
        # It is highly robust for 1D scalar optimization on non-smooth topologies.

        # 'bounded' requires finite bounds. If upper_bound is None, we default
        # to a generous limit (e.g., 100.0) to account for extreme dynamic range
        # mismatches without searching infinitely.
        brent_upper = upper_bound if upper_bound is not None else 100.0

        # If the bound is effectively zero, skip the heavy lifting
        if brent_upper <= 0.0:
            return 0.0

        # Ensure init_alpha is valid
        if init_alpha > brent_upper:
            init_alpha = brent_upper
        elif init_alpha < 0.0:
            init_alpha = 0.0

        # We use bounded Brent search for robust bracketing of non-smooth topologies
        result = minimize_scalar(
            func,
            bounds=(0.0, brent_upper),
            method='bounded',
            options={'xatol': 1e-4, 'maxiter': 100}
        )

        if not result.success:
            print(f"Warning: Brent minimization failed to converge: {result.message}", file=sys.stdout)

        opt_alpha = float(result.x)

        # We generally do not revert to 0 unless it's drastically worse because
        # Brent is rigorous, but for legacy backwards compatibility with some
        # delicate test tolerances, we bound it if it goes very wrong.
        if opt_alpha > 1e-6:
            cost_zero = func(0.0)
            cost_opt = func(opt_alpha)
            if cost_opt > cost_zero:
                opt_alpha = 0.0

        return opt_alpha

    elif method == 'cobyla':
        # COBYLA: Multivariate optimizer applied to a 1D problem.
        # Retained for legacy comparisons and benchmarking.
        cons = [lambda a: a]  # constraint: alpha >= 0
        if upper_bound is not None:
            cons.append(lambda a, ub=upper_bound: ub - a)

        # Ensure init_alpha is within bounds to prevent immediate constraint violation
        start_alpha = init_alpha
        if upper_bound is not None and start_alpha > upper_bound:
            start_alpha = upper_bound

        # Heuristic step size based on bin width
        rhobeg = 1.0 / bins

        result_array: npt.NDArray = fmin_cobyla(
            func=func,
            x0=np.array([start_alpha]),
            cons=cons,
            rhobeg=rhobeg,
            rhoend=1e-8,
            disp=0  # Suppress internal scipy printing
        )
        return float(result_array.item())

    else:
        raise ValueError(f"Unknown optimization method: '{method}'. Use 'brent' or 'cobyla'.")


def compute_unmixing_matrix(
    images: Sequence[npt.NDArray],
    *,
    downscale: Union[int, tuple[int, ...], None] = None,
    max_iters=1_000,
    step_mult=0.1,
    verbose=False,
    return_iters=False,
    max_samples: Union[float, int] = 100_000,
    min_samples: int = 1_000,
    quantile: float = 0.95,
    theoretical_mixing_matrix: Union[npt.NDArray, None] = None,
    method: Literal['brent', 'cobyla'] = 'cobyla',
) -> npt.NDArray:
    n_channels = len(images)

    if downscale is not None:
        if isinstance(downscale, int):
            factors = tuple(downscale for _ in range(images[0].ndim))
        else:
            factors = downscale

        if verbose:
            print(f"Downscaling images by factors {factors} before pixel selection...", file=sys.stdout)

        processed_images = [_downscale_local_mean(img, factors) for img in images]
    else:
        processed_images = images

    # Select representative pixels for unmixing optimization
    image_flat = select_representative_pixels(
        processed_images,
        quantile=quantile,
        max_samples=max_samples,
        min_samples=min_samples,
        verbose=verbose,
    )

    image_orig = image_flat.copy()
    image = image_flat

    n_samples = image_flat.shape[1]
    optimal_bins = compute_optimal_bins(n_samples)

    if verbose:
        print(f"Using {optimal_bins} bins for mutual information estimation (N={n_samples})", file=sys.stdout)

    mat_cumul = np.eye(n_channels, dtype=float)

    mats = []
    for _ in tqdm(
        range(max_iters),
        disable=not verbose,
        desc="Unmixing iterations",
        total=max_iters,
        file=sys.stdout,
    ):
        mat = np.eye(n_channels, dtype=float)

        for row in range(n_channels):
            for col in range(n_channels):
                if row == col:
                    continue

                init_alpha = 0.0
                upper_bound = None

                if theoretical_mixing_matrix is not None:
                    # M_theo[row, col] represents bleed-through from dye `col` into channel `row`
                    theo_bleed = theoretical_mixing_matrix[row, col]
                    if theo_bleed == 0.0:
                        upper_bound = 0.0
                    else:
                        # Allow some flexibility, but use theory as a strong initialization
                        # or bound. If the theoretical bleed-through is very small, we might
                        # constrain it, but for now we just use it as initialization if it's
                        # the first iteration, and optionally as an upper bound.
                        # Since we are solving iteratively for residual mixing,
                        # init_alpha should be 0.0 for residual passes.

                        # We use the theoretical matrix to strictly bound unmixing:
                        # If theory says maximum bleed-through is X, we don't allow unmixing
                        # coefficient to exceed X in total. Since we iteratively build mat_cumul,
                        # this is slightly tricky. For a simple implementation, if theory == 0,
                        # bound = 0.
                        pass

                # Initialize alpha to 0.0 because we are estimating residual mixing
                coef = minimize_mi(
                    image[col],
                    image[row],
                    init_alpha=init_alpha,
                    bins=optimal_bins,
                    upper_bound=upper_bound,
                    method=method,
                )
                mat[row, col] = -step_mult * coef

        # Convergence check: check if the update matrix is close to identity
        # This implies that all off-diagonal updates (coefs) were small.
        if np.allclose(mat, np.eye(n_channels)):
            break

        # If theoretical_mixing_matrix is provided, ensure cumulative matrix doesn't exceed theory
        # BEFORE we commit the update to mat_cumul.
        # We compute mat_cumul_next to clamp it.
        mat_cumul_next = mat @ mat_cumul

        # constrain coefficients to 1.0 along the diagonal, and negative for
        # off-diagonal entries
        for row in range(n_channels):
            for col in range(n_channels):
                if row == col:
                    mat_cumul_next[row, col] = 1.0
                else:
                    if mat_cumul_next[row, col] > 0.0:
                        mat_cumul_next[row, col] = 0.0

        if theoretical_mixing_matrix is not None:
            # We want to bound U so it doesn't unmix more than the mathematical limit.
            # For a 2x2 matrix M with 1s on diag, M^-1 has off-diagonals bounded by -M_ij
            # when normalized. So U_ij >= -M_theo_ij.
            # But we must compute the exact U from M_theo.

            # Simple per-element constraint works well:
            for row in range(n_channels):
                for col in range(n_channels):
                    if row != col:
                        theo_bleed = theoretical_mixing_matrix[row, col]
                        if theo_bleed == 0.0:
                            mat_cumul_next[row, col] = 0.0
                        else:
                            if theo_bleed >= 0:
                                if mat_cumul_next[row, col] < -theo_bleed:
                                    mat_cumul_next[row, col] = -theo_bleed

        mat_cumul = mat_cumul_next
        mats.append(mat_cumul.copy())

        # update the next iteration of image
        assert mat_cumul.shape == (n_channels, n_channels)
        # several times faster than np.einsum
        image = np.tensordot(mat_cumul, image_orig, axes=1)

    if not mats:
        mats.append(np.eye(n_channels, dtype=float))

    if return_iters:
        return np.stack(mats)
    else:
        return mats[-1]


def apply_unmixing_matrix(
    images: Sequence[npt.NDArray], matrix: npt.NDArray
) -> list[npt.NDArray]:
    """
    Apply unmixing matrix to a sequence of images of arbitrary dimensions.

    Parameters
    ----------
    images : Sequence[npt.NDArray]
        A sequence of image arrays, one per channel. Each array should have the same shape.
    matrix : ndarray
        The unmixing matrix of shape (C, C).

    Returns
    -------
    unmixed_images : list[npt.NDArray]
        A list of unmixed image arrays, one per channel.
    """
    if matrix.shape[1] != len(images):
        raise ValueError(
            f"Matrix input channels ({matrix.shape[1]}) must match number of image "
            f"channels ({len(images)})"
        )

    try:
        image = np.stack(images, axis=0)
    except MemoryError:
        raise MemoryError("Not enough memory to stack the input images. Please provide smaller images or fewer channels.")

    unmixed = np.tensordot(matrix, image, axes=1)

    # Return as a list of individual channel arrays
    return [unmixed[i] for i in range(unmixed.shape[0])]
