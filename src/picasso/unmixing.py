import math
import sys

import numpy as np
import numpy.typing as npt
from fast_histogram import histogram2d, histogram1d
from scipy.optimize import fmin_cobyla
from skimage.util import img_as_float
from tqdm import tqdm


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


def minimize_mi(x: npt.NDArray, y: npt.NDArray, *, init_alpha=0.0) -> float:
    def func(alpha: npt.NDArray):
        return mutual_information(x, y - alpha * x)

    result: npt.NDArray = fmin_cobyla(
        func=func,
        x0=np.array([init_alpha]),
        cons=[lambda a: a],
        rhobeg=1e-2,
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
    max_samples=100_000,
) -> npt.NDArray:
    n_channels = image.shape[0]

    image = img_as_float(image)
    image_flat = image.reshape(n_channels, -1)

    if max_samples is not None and image_flat.shape[1] > max_samples:
        n_pixels = image_flat.shape[1]

        # If image is massive, we can't afford to compute norms on all pixels.
        # Heuristic: if N > 10 * max_samples, sample a subset first
        if n_pixels > 10 * max_samples:
            subset_indices = np.random.randint(0, n_pixels, 10 * max_samples)
            candidate_pixels = image_flat[:, subset_indices]
        else:
            candidate_pixels = image_flat

        # Compute intensities (L2 norm)
        intensities = np.linalg.norm(candidate_pixels, axis=0)

        # Filter out very low intensity (background)
        mask = intensities > 1e-6
        if np.any(mask):
            candidate_pixels = candidate_pixels[:, mask]
            intensities = intensities[mask]

        # Select max_samples randomly from valid pixels
        if candidate_pixels.shape[1] > max_samples:
            indices = np.random.choice(
                candidate_pixels.shape[1], max_samples, replace=False
            )
            image_flat = candidate_pixels[:, indices]
        else:
            image_flat = candidate_pixels

        if verbose:
            print(
                f"Subsampled to {image_flat.shape[1]} pixels.", file=sys.stdout
            )

    image_orig = image_flat.copy()
    image = image_flat

    mat_cumul = np.eye(n_channels, dtype=float)
    mat_last = np.eye(n_channels, dtype=float)

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

                coef = minimize_mi(
                    image[col], image[row], init_alpha=mat_last[row, col]
                )
                mat[row, col] = -step_mult * coef

        # check this early on
        if np.allclose(mat, mat_last):
            break
        mat_last = mat.copy()

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
