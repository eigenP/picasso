import numpy as np

from picasso.spectra import (
    interpolate_spectrum,
    load_fpbase_spectra,
    standardize_spectra,
)

def test_interpolate_spectrum():
    wl = np.array([400, 410, 420])
    intensities = np.array([0, 100, 0])
    new_wl = np.array([390, 400, 405, 410, 415, 420, 430])

    interp = interpolate_spectrum(wl, intensities, new_wl)

    assert np.allclose(interp, [0, 0, 50, 100, 50, 0, 0])

def test_standardize_spectra():
    wl1 = np.array([400, 410, 420])
    int1 = np.array([0, 100, 0])

    wl2 = np.array([500, 510, 520])
    int2 = np.array([0, 200, 0])

    new_wl, standardized = standardize_spectra([(wl1, int1), (wl2, int2)], start_wl=300, end_wl=600, step=10)

    assert new_wl.shape[0] == 31
    assert standardized[0].max() == 1.0
    assert standardized[1].max() == 1.0

    assert standardized[0][11] == 1.0 # 410 nm
    assert standardized[1][21] == 1.0 # 510 nm

    assert np.sum(standardized[0] > 0) == 1 # only 410 is non-zero in the interpolated array since 400 and 420 are 0
