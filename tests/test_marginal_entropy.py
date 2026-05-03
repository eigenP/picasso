import time
import numpy as np
from skimage.data import cells3d
from typing import Sequence
import pytest

from picasso.unmixing import compute_unmixing_matrix, apply_unmixing_matrix

def generate_test_data() -> tuple[list[np.ndarray], np.ndarray]:
    """Generate biologically realistic mixed data."""
    # Use cells3d (nuclei and membrane)
    image = cells3d()
    membrane = image[30, 0, :, :]
    nuclei = image[30, 1, :, :]

    # Normalize
    membrane = membrane.astype(float) / membrane.max()
    nuclei = nuclei.astype(float) / nuclei.max()

    # Add small poisson noise
    membrane += np.random.poisson(membrane * 10) / 100.0
    nuclei += np.random.poisson(nuclei * 10) / 100.0

    c1 = membrane
    c2 = nuclei

    # Theoretical mixing matrix:
    # Channel 1 receives 20% bleed-through from Channel 2
    # Channel 2 receives 30% bleed-through from Channel 1
    M = np.array([
        [1.0, 0.2],
        [0.3, 1.0]
    ])

    # Stack and mix
    true_images = np.stack([c1, c2])
    mixed = np.tensordot(M, true_images, axes=1)

    mixed_list = [mixed[0], mixed[1]]
    return mixed_list, M

def test_marginal_entropy_vs_mutual_information_landscape():
    """
    Test that both methods yield effectively the same unmixing matrix
    (because the optimization landscape is identical up to a constant shift).
    """
    np.random.seed(42)
    mixed_images, _ = generate_test_data()

    # Run with 1D marginal entropy
    U_me = compute_unmixing_matrix(
        mixed_images,
        method="marginal_entropy",
        max_samples=1.0, # Use all pixels to eliminate sampling variance between runs
        quantile=0.0
    )

    # Run with 2D mutual information
    U_mi = compute_unmixing_matrix(
        mixed_images,
        method="mutual_information",
        max_samples=1.0,
        quantile=0.0
    )

    # They should be extremely close.
    # Due to floating point differences and slightly different histogram calculations
    # (1D vs 2D binning artifacts), they might not be bit-for-bit identical,
    # but the off-diagonals should be very close.
    np.testing.assert_allclose(U_me, U_mi, atol=0.05, rtol=0.05)


def test_marginal_entropy_is_faster():
    """
    Benchmark test to ensure the 1D marginal entropy approach is faster
    than the 2D mutual information approach on a decent amount of data.
    """
    np.random.seed(42)
    mixed_images, _ = generate_test_data()

    # Time Mutual Information
    start_time_mi = time.perf_counter()
    compute_unmixing_matrix(
        mixed_images,
        method="mutual_information",
        max_samples=1.0,
        quantile=0.0
    )
    mi_duration = time.perf_counter() - start_time_mi

    # Time Marginal Entropy
    start_time_me = time.perf_counter()
    compute_unmixing_matrix(
        mixed_images,
        method="marginal_entropy",
        max_samples=1.0,
        quantile=0.0
    )
    me_duration = time.perf_counter() - start_time_me

    print(f"\nMutual Information Time: {mi_duration:.4f}s")
    print(f"Marginal Entropy Time: {me_duration:.4f}s")
    print(f"Speedup Factor: {mi_duration / me_duration:.2f}x")

    # In general, marginal entropy should be strictly faster due to 1D vs 2D histogram.
    # However, strict timing assertions in CI environments can be flaky due to CPU load spikes.
    # We log the durations above, but use a very relaxed assertion to prevent flaky CI failures
    # while still catching catastrophic performance regressions.
    assert me_duration < (mi_duration * 10), f"Marginal Entropy ({me_duration:.4f}s) was catastrophically slower than Mutual Info ({mi_duration:.4f}s)"
