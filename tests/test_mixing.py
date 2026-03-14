import numpy as np
from picasso.spectra import standardize_spectra
from picasso.mixing import compute_theoretical_mixing_matrix

def test_compute_theoretical_mixing_matrix():
    # Setup simple synthetic spectra
    wl1 = np.array([400, 450, 500])
    int1 = np.array([0, 100, 0]) # peak at 450

    wl2 = np.array([450, 500, 550])
    int2 = np.array([0, 100, 0]) # peak at 500

    # Standardize them
    wavelengths, standardized = standardize_spectra(
        [(wl1, int1), (wl2, int2)], start_wl=400, end_wl=550, step=10
    )

    # Channel 1: 440-460 (primarily captures dye 1, peak 450)
    # Channel 2: 490-510 (primarily captures dye 2, peak 500)
    collection_bands = [(440, 460), (490, 510)]

    # Compute theoretical mixing matrix
    M = compute_theoretical_mixing_matrix(wavelengths, standardized, collection_bands)

    assert M.shape == (2, 2)
    # Since the spectra have compact support and don't overlap in the bands:
    assert M[0, 0] == 1.0
    assert M[1, 1] == 1.0
    assert np.allclose(M[0, 1], 0.0, atol=0.1) # dye 2 has no intensity in 440-460
    assert np.allclose(M[1, 0], 0.0, atol=0.1) # dye 1 has no intensity in 490-510

    # Now let's test with bleed-through:
    # Channel 1: 400-500 (captures dye 1, some dye 2)
    # Channel 2: 480-550 (captures dye 2, some dye 1)

    # Let's use broader spectra for a better test
    wl1_broad = np.array([400, 450, 500, 550])
    int1_broad = np.array([10, 100, 50, 0]) # Bleeds into 500+

    wl2_broad = np.array([400, 450, 500, 550])
    int2_broad = np.array([0, 20, 100, 10]) # Bleeds into 450

    wavelengths, standardized = standardize_spectra(
        [(wl1_broad, int1_broad), (wl2_broad, int2_broad)], start_wl=400, end_wl=550, step=10
    )

    collection_bands_broad = [(400, 470), (480, 550)]

    M_broad = compute_theoretical_mixing_matrix(wavelengths, standardized, collection_bands_broad)

    # M_broad[0, 0] is sum(dye 1 in 400-470), normalized to itself = 1.0
    # M_broad[1, 1] is sum(dye 2 in 480-550), normalized to itself = 1.0
    assert M_broad[0, 0] == 1.0
    assert M_broad[1, 1] == 1.0

    # Dye 1 bleeds into Channel 2 (480-550)
    assert M_broad[1, 0] > 0.0

    # Dye 2 bleeds into Channel 1 (400-470)
    assert M_broad[0, 1] > 0.0
