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

@pytest.mark.parametrize("dye1_max, dye2_max, expected_max_error", [
    (1_000, 1_000, 0.05),     # 1:1 ratio. (Error higher here sometimes due to binning/step size)
    (10_000, 1_000, 0.03),    # 10:1 ratio.
    (50_000, 500, 0.03),      # 100:1 ratio. Noise in Ch2 bleed (sqrt(5000) ~ 70) is smaller than Dye 2 (500).
])
def test_floodlight_and_candle(dye1_max, dye2_max, expected_max_error):
    """
    Test 1: The 'Floodlight and Candle' Test (Tests SNR limits)
    Setup: Generate a synthetic image sweeping the ratio of Dye 1 to Dye 2 intensities.
    Mix them with 10% bleed-through. Add Poisson noise.
    """
    membrane, nuclei = get_biological_structures()

    # Scale structures: Floodlight (Dye 1) and Candle (Dye 2)
    dye1 = membrane / membrane.max() * dye1_max
    dye2 = nuclei / nuclei.max() * dye2_max

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

    # Unconstrained unmixing
    U = compute_unmixing_matrix(
        list(mixed_noisy),
        max_iters=50,
        quantile=0.0,  # Use all pixels to help with noise
        step_mult=0.1
    )

    # Check if we recovered the bleed-through coefficient
    error = abs(U[1, 0] - (-true_bleed))
    assert error <= expected_max_error, f"Expected success at {dye1_max}:{dye2_max} but error was {error:.4f}"

@pytest.mark.xfail(reason="Tests SNR limit. Algorithm struggles due to noise overpowering correlation.")
@pytest.mark.parametrize("dye1_max, dye2_max", [
    (50_000, 10),      # 5000:1 ratio. Noise (70) completely buries Dye 2 (10). Correlation is lost in noise.
    (100_000, 10),     # 10000:1 ratio.
])
def test_floodlight_and_candle_failures(dye1_max, dye2_max):
    membrane, nuclei = get_biological_structures()
    dye1 = membrane / membrane.max() * dye1_max
    dye2 = nuclei / nuclei.max() * dye2_max
    true_bleed = 0.10
    M_true = np.array([
        [1.0, 0.0],
        [true_bleed, 1.0]
    ])
    sources = np.stack([dye1, dye2])
    mixed = np.tensordot(M_true, sources, axes=1)
    mixed_noisy = add_poisson_noise(mixed)
    U = compute_unmixing_matrix(
        list(mixed_noisy),
        max_iters=50,
        quantile=0.0,
        step_mult=0.1
    )
    error = abs(U[1, 0] - (-true_bleed))
    # Asserting it strictly fails to find the correlation
    assert error > 0.05, f"Algorithm miraculously succeeded at SNR limit with error {error:.4f}"


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
        list(mixed),
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
        list(mixed),
        max_iters=50,
        quantile=0.5,
        theoretical_mixing_matrix=M_theo
    )

    # Constrained should hit the wall at 10% exactly and stop.
    # U_constrained[1, 0] should be around -0.10. Since we started with 0 initialization
    # and moved slowly, it will bound tightly to the theoretical constraint without over-unmixing.
    assert np.isclose(U_constrained[1, 0], -optical_bleed, atol=0.05), "Constrained unmixing did not bound at M_theo."


def test_real_spectra_theoretical_bound():
    """
    Test 4: Using "real" synthetic spectra to generate theoretical mixing bound.
    Setup: Uses AF 546 and AF 594 approximate spectra properties, defines two collection channels
    (550-590nm and 596-640nm). Computes M_theo and ensures the algorithm uses it properly.
    """
    from picasso.spectra import standardize_spectra
    from picasso.mixing import compute_theoretical_mixing_matrix

    # Mocking AF 546 and AF 594 spectra (gaussian approx)
    w = np.arange(400, 800, 1.0)
    # AF 546 peak ~ 555, AF 594 peak ~ 610
    i_546 = np.exp(-0.5 * ((w - 555) / 20) ** 2)
    i_594 = np.exp(-0.5 * ((w - 610) / 25) ** 2)

    # standardize_spectra expects List[Tuple[npt.NDArray, npt.NDArray]]
    spectra = [
        (w, i_546),
        (w, i_594)
    ]

    new_wl, std_spectra = standardize_spectra(spectra, start_wl=400, end_wl=800, step=1.0)

    # Collection bands for typical AF 546 / AF 594 configuration
    # Ch1 (AF 546): 550 - 590
    # Ch2 (AF 594): 596 - 640
    collection_bands = [(550, 590), (596, 640)]

    M_theo = compute_theoretical_mixing_matrix(
        new_wl, std_spectra, collection_bands
    )

    # Calculate M_theo properties
    # Expect Dye 1 to bleed slightly into Ch 2
    # Expect Dye 2 to have minimal/zero bleed into Ch 1

    membrane, nuclei = get_biological_structures()

    # Using the exact mathematical M_theo to create the mixed image
    dye1 = membrane / membrane.max() * 5_000
    dye2 = nuclei / nuclei.max() * 5_000
    sources = np.stack([dye1, dye2])

    mixed = np.tensordot(M_theo, sources, axes=1)
    mixed_noisy = add_poisson_noise(mixed)

    # Unmix using the exact same M_theo
    U_constrained = compute_unmixing_matrix(
        list(mixed_noisy),
        max_iters=50,
        quantile=0.5,
        theoretical_mixing_matrix=M_theo
    )

    # We assert that the recovered matrix stays within the theoretical bleed bounds.
    bleed1_into_2 = M_theo[1, 0]
    bleed2_into_1 = M_theo[0, 1]

    # If M_theo says max bleed1->2 is X, U[1, 0] cannot be less than -X.
    # U_constrained[1, 0] should be approx -bleed1_into_2
    assert U_constrained[1, 0] >= -bleed1_into_2 - 0.02
    assert U_constrained[0, 1] >= -bleed2_into_1 - 0.02


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
        list(mixed),
        max_iters=50,
        quantile=0.5,
        theoretical_mixing_matrix=M_theo_bad
    )

    # The algorithm should try to unmix (because true is 15%), but hits the 5% wall.
    assert np.isclose(U_bad_physics[1, 0], -bad_theo_bleed, atol=0.01), "Did not stop at the bad physics limit."
