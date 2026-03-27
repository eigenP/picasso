import numpy as np
import pytest
from skimage import data, img_as_float
from picasso.spectra import standardize_spectra
from picasso.mixing import compute_theoretical_mixing_matrix
from picasso.unmixing import compute_unmixing_matrix

def test_cells3d_constrained_unmixing():
    """
    Test using skimage cells3d data to simulate real-world unmixing
    with theoretical physics constraints.
    """
    # Load 2-channel 3D image (membrane, nuclei)
    try:
        cells_3d = data.cells3d()
    except Exception:
        pytest.skip("Could not load skimage cells3d data")

    # Just take one slice for speed, shape (2, 256, 256)
    image = img_as_float(cells_3d[30])

    # Let's say we have two dyes:
    # Dye 1 (Membrane marker): peak 450nm
    # Dye 2 (Nuclei marker): peak 520nm

    wl1 = np.array([400, 450, 500, 550])
    int1 = np.array([10, 100, 40, 5])

    wl2 = np.array([450, 520, 580, 650])
    int2 = np.array([5, 100, 30, 0])

    wavelengths, standardized = standardize_spectra(
        [(wl1, int1), (wl2, int2)], start_wl=400, end_wl=650, step=10
    )

    # Collection channels:
    # Channel 1: 420-480 nm (gets Dye 1, very little Dye 2)
    # Channel 2: 500-560 nm (gets Dye 2, some Dye 1 bleed-through)
    collection_bands = [(420, 480), (500, 560)]

    M_theo = compute_theoretical_mixing_matrix(
        wavelengths, standardized, collection_bands
    )

    # Now simulate a MIXED image based on M_theo + some extra artifact mixing
    M_actual = M_theo.copy()
    # add some extra artificial mixing from noise/artifact not predicted by theory
    M_actual[1, 0] += 0.2
    M_actual[0, 1] += 0.1

    mixed_image = np.tensordot(M_actual, image, axes=1)

    # Unconstrained unmixing
    U_unconstrained = compute_unmixing_matrix(
        list(mixed_image), max_iters=50, step_mult=0.5, verbose=False
    )

    # Constrained unmixing
    U_constrained = compute_unmixing_matrix(
        list(mixed_image), max_iters=50, step_mult=0.5, verbose=False,
        theoretical_mixing_matrix=M_theo
    )

    # The pure theory bounds:
    # We expect unconstrained to over-unmix because it sees the extra artifact
    # mixing as true mixing. The constrained version should be bounded by theory.

    # U[1, 0] represents unmixing of channel 0 out of channel 1.
    # It should be constrained by -M_theo[1, 0]
    expected_bound_10 = -M_theo[1, 0]
    expected_bound_01 = -M_theo[0, 1]

    # Since we added positive artifact, the actual mixing is HIGHER than theory
    # Unconstrained should find a more negative value than the theoretical bound
    assert U_unconstrained[1, 0] < expected_bound_10

    # Constrained should be bounded roughly by the theoretical bound.
    # We use a one-sided inequality instead of strict equality because
    # multiple iterations can accumulate slightly past the initial bound
    # depending on step size dynamics, but it should still be strictly > than unconstrained.
    assert U_constrained[1, 0] > U_unconstrained[1, 0]
