import math
import sys
from typing import Union

import numpy as np
import numpy.typing as npt
from fast_histogram import histogram2d, histogram1d
from scipy.optimize import fmin_cobyla
from skimage.util import img_as_float
from tqdm import tqdm


def select_representative_pixels(
    image: npt.NDArray,
    quantile: float = 0.95,
    max_samples: Union[float, int] = 100_000,
    min_samples: int = 1_000,
    verbose: bool = False,
) -> npt.NDArray:
    n_channels = image.shape[0]
    n_pixels = image[0].size

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
        # Interpret as a fraction of the ORIGINAL total pixels
        # (or maybe valid pixels? User intent is likely "fraction of image")
        target_samples = int(max_samples * n_pixels)
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
        indices = np.random.choice(
            n_high_signal, target_samples, replace=False
        )
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


def shannon_entropy(a: npt.NDArray) -> float:
    """
    This runs very often, so we should do our best to make this fast.
    See 'profiling/shannon-entropy.ipynb' for why this was chosen
    """
    # seems like flattened arrays are faster, but .flatten() produces a copy so we
    # stick with .ravel()
    a = a.ravel()
    a /= a.sum()
    a = a[np.nonzero(a != 0)]
    a *= np.log2(a)
    return -a.sum().item()


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


def minimize_mi(x: npt.NDArray, y: npt.NDArray, *, init_alpha=0.0, bins=100) -> float:
    def func(alpha: npt.NDArray):
        return mutual_information(x, y - alpha * x, bins=bins)

    # Heuristic: set initial step size (rhobeg) based on bin width relative to range.
    # Assuming range is roughly constant and x is significant, a change in alpha
    # needs to shift values by at least one bin width to change the histogram counts.
    # bin_width ~ Range / bins.
    # delta_y = delta_alpha * x. We need delta_y ~ bin_width.
    # So delta_alpha ~ Range / (bins * x).
    # Assuming Range/x is roughly order of 1, rhobeg should scale with 1/bins.
    # For 100 bins, 1/100 = 0.01, which matches the original default.
    rhobeg = 1.0 / bins

    result: npt.NDArray = fmin_cobyla(
        func=func,
        x0=np.array([init_alpha]),
        cons=[lambda a: a],
        rhobeg=rhobeg,
        rhoend=1e-8,
    )
    return result.item()


def compute_unmixing_matrix(
    image: npt.NDArray,
    *,
    max_iters=1_000,
    step_mult=0.1,
    verbose=False,
    return_iters=False,
    max_samples: Union[float, int] = 100_000,
    min_samples: int = 1_000,
    quantile: float = 0.95,
) -> npt.NDArray:
    n_channels = image.shape[0]

    # Select representative pixels for unmixing optimization
    image_flat = select_representative_pixels(
        image,
        quantile=quantile,
        max_samples=max_samples,
        min_samples=min_samples,
        verbose=verbose,
    )

    image_orig = image_flat.copy()
    image = image_flat

    # Adaptive binning based on sample size
    # Rule of thumb: scaled by cubic root of N, relative to 100 bins for 100k samples
    # For 100k samples: (100k/100k)^(1/3) * 100 = 100 bins
    # For 1k samples: (1k/100k)^(1/3) * 100 = 0.215 * 100 = 21 bins
    # For 89 samples: (89/100k)^(1/3) * 100 = 0.096 * 100 = 9 bins
    n_samples = image_flat.shape[1]
    optimal_bins = int(100 * (n_samples / 100_000) ** (1 / 3))
    optimal_bins = max(5, min(100, optimal_bins))

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

                # Initialize alpha to 0.0 because we are estimating residual mixing
                coef = minimize_mi(
                    image[col], image[row], init_alpha=0.0, bins=optimal_bins
                )
                mat[row, col] = -step_mult * coef

        # Convergence check: check if the update matrix is close to identity
        # This implies that all off-diagonal updates (coefs) were small.
        if np.allclose(mat, np.eye(n_channels)):
            break

        # update matrix
        assert mat_cumul.shape == (n_channels, n_channels)
        mat_cumul = mat @ mat_cumul

        # constrain coefficients to 1.0 along the diagonal, and negative for
        # off-diagonal entries
        for row in range(n_channels):
            for col in range(n_channels):
                if row == col:
                    mat_cumul[row, col] = 1.0
                else:
                    if mat_cumul[row, col] > 0.0:
                        mat_cumul[row, col] = 0.0
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
    image: npt.NDArray, matrix: npt.NDArray
) -> npt.NDArray:
    """
    Apply unmixing matrix to an image of arbitrary dimensions.

    Parameters
    ----------
    image : ndarray
        The input image of shape (C, ...).
    matrix : ndarray
        The unmixing matrix of shape (C, C).

    Returns
    -------
    unmixed_image : ndarray
        The unmixed image of shape (C, ...).
    """
    if matrix.shape[1] != image.shape[0]:
        raise ValueError(
            f"Matrix input channels ({matrix.shape[1]}) must match image "
            f"channels ({image.shape[0]})"
        )
    return np.tensordot(matrix, image, axes=1)
