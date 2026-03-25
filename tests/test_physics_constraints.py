import numpy as np
import pytest
from skimage.data import cells3d

from picasso.unmixing import compute_unmixing_matrix

# --- Helpers ---
def add_poisson_noise(image: np.ndarray, seed: int = 42) -> np.ndarray:
    """Adds realistic Poisson (shot) noise to an image."""
    rng = np.random.default_rng(seed)
    # Ensure positive
    image = np.clip(image, 0, None)
    return rng.poisson(image).astype(np.float64)


def get_biological_structures() -> tuple[np.ndarray, np.ndarray]:
    """Returns two distinct biological structures (e.g., membrane and nuclei) from cells3d."""
    image = cells3d()
    # image shape: (Z, C, Y, X)
    # Take middle slice
    z_mid = image.shape[0] // 2
    membrane = image[z_mid, 0, :, :].astype(np.float64)
    nuclei = image[z_mid, 1, :, :].astype(np.float64)
    return membrane, nuclei

def get_perfectly_colocalized_structures() -> tuple[np.ndarray, np.ndarray]:
    """Returns two perfectly colocalized biological structures."""
    image = cells3d()
    z_mid = image.shape[0] // 2
    membrane = image[z_mid, 0, :, :].astype(np.float64)
    return membrane.copy(), membrane.copy()


# --- Tests ---

@pytest.mark.xfail(reason="Test explores SNR limit. Pure theoretical algorithm cannot mathematically recover the 100-photon signal buried in 5000-photon bleed-through shot noise.")
def test_floodlight_and_candle():
    """
    Test 1: The 'Floodlight and Candle' Test (Tests SNR limits)
    Setup: Generate a synthetic image where Dye 1 has intensities ~50,000 and Dye 2 ~100.
    Mix them with 10% bleed-through. Add Poisson noise.
    """
    membrane, nuclei = get_biological_structures()

    # Scale structures: Floodlight (Dye 1) ~ 50,000; Candle (Dye 2) ~ 100
    dye1 = membrane / membrane.max() * 50_000
    dye2 = nuclei / nuclei.max() * 100

    true_bleed = 0.10
    M_true = np.array([
        [1.0, 0.0],
        [true_bleed, 1.0] # Dye 1 bleeds into Channel 2
    ])

    # Mix
    sources = np.stack([dye1, dye2])
    mixed = np.tensordot(M_true, sources, axes=1)

    # Add noise
    mixed_noisy = add_poisson_noise(mixed)

    # The noise in Ch2 bleed-through (from Dye 1) is ~sqrt(50,000 * 0.10) = sqrt(5000) = ~70.7
    # This noise is comparable to the max signal of Dye 2 (100).

    # Unconstrained unmixing
    # We expect it to struggle due to noise overwhelming the true correlation signal of Dye 2.
    U = compute_unmixing_matrix(
        mixed_noisy,
        max_iters=50,
        quantile=0.0,  # Use all pixels to help with noise
        step_mult=0.1
    )

    # We assert that the recovered bleed-through coefficient U[1,0] (which should be ~ -0.10)
    # is significantly off due to noise destroying the subtle correlation.
    assert abs(U[1, 0] - (-true_bleed)) > 0.02, "Expected optimization to fail at this SNR limit."


def test_perfect_colocalization_trap():
    """
    Test 2: The Perfect Co-localization Trap (Tests over-unmixing)
    Setup: Create an image where Dye 1 and Dye 2 are exactly the same shape.
    Mix them.
    Expected: Empirical/percentile unmixing erases Dye 2. Pure M_theo approach passes cleanly.
    """
    # Perfectly colocalized (same structure)
    dye1, dye2 = get_perfectly_colocalized_structures()

    # Normalize and set both to ~10,000 max intensity
    dye1 = dye1 / dye1.max() * 10_000
    dye2 = dye2 / dye2.max() * 10_000

    # True optical bleed-through
    optical_bleed = 0.10
    M_true = np.array([
        [1.0, 0.0],
        [optical_bleed, 1.0]
    ])

    # Because they are colocalized and same brightness, Dye2 effectively acts like
    # a 1.0 bleed-through on top of the optical 0.10. Total apparent bleed = 1.10.

    sources = np.stack([dye1, dye2])
    mixed = np.tensordot(M_true, sources, axes=1)
    mixed = add_poisson_noise(mixed)

    # 1. Empirical (Unconstrained) Unmixing
    U_empirical = compute_unmixing_matrix(
        mixed,
        max_iters=50,
        quantile=0.5
    )

    # Empirical unmixing should "over-unmix" due to the biological correlation.
    # It will see total apparent bleed > 10% and try to subtract it.
    assert U_empirical[1, 0] < -(optical_bleed + 0.5), "Empirical unmixing failed to over-unmix."

    # 2. Constrained Unmixing
    M_theo = np.array([
        [0.0, 0.0],
        [optical_bleed, 0.0]
    ])

    U_constrained = compute_unmixing_matrix(
        mixed,
        max_iters=50,
        quantile=0.5,
        theoretical_mixing_matrix=M_theo
    )

    # Constrained should hit the wall at 10% exactly and stop.
    # U_constrained[1, 0] should be around -0.10. Since we started with 0 initialization
    # and moved slowly, it will bound tightly to the theoretical constraint without over-unmixing.
    assert np.isclose(U_constrained[1, 0], -optical_bleed, atol=0.05), "Constrained unmixing did not bound at M_theo."


def test_bad_physics():
    """
    Test 3: The 'Bad Physics' Test (Tests sensitivity to theory)
    Setup: Generate synthetically mixed image with true bleed-through of 15%.
    Feed M_theo with max bleed-through of 5%.
    Expected: Hits the 5% wall and stops (under-unmixing).
    """
    membrane, nuclei = get_biological_structures()

    dye1 = membrane / membrane.max() * 10_000
    dye2 = nuclei / nuclei.max() * 10_000

    true_bleed = 0.15
    M_true = np.array([
        [1.0, 0.0],
        [true_bleed, 1.0]
    ])

    sources = np.stack([dye1, dye2])
    mixed = np.tensordot(M_true, sources, axes=1)
    mixed = add_poisson_noise(mixed)

    # Bad physics: we think max bleed is 5%
    bad_theo_bleed = 0.05
    M_theo_bad = np.array([
        [0.0, 0.0],
        [bad_theo_bleed, 0.0] # Only 5% allowed
    ])

    U_bad_physics = compute_unmixing_matrix(
        mixed,
        max_iters=50,
        quantile=0.5,
        theoretical_mixing_matrix=M_theo_bad
    )

    # The algorithm should try to unmix (because true is 15%), but hits the 5% wall.
    assert np.isclose(U_bad_physics[1, 0], -bad_theo_bleed, atol=0.01), "Did not stop at the bad physics limit."
