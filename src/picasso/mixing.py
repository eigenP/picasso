import numpy as np
import numpy.typing as npt
from typing import List, Tuple

def compute_theoretical_mixing_matrix(
    wavelengths: npt.NDArray,
    standardized_spectra: List[npt.NDArray],
    collection_bands: List[Tuple[float, float]],
) -> npt.NDArray:
    """
    Compute a theoretical mixing matrix from standardized spectra and collection bands.

    This function calculates the expected relative brightness of each dye in each
    collection channel by integrating the dye's emission spectrum over the channel's
    collection band.

    The resulting matrix M will have shape (n_channels, n_dyes), where M[i, j]
    is the relative brightness of dye j in channel i.

    Each column j of M (the dye's bleed-through profile) is normalized such that
    the "primary" channel for that dye is 1.0. We assume the primary channel for
    dye j is channel j. This means M has 1.0 on its diagonal, and off-diagonal
    elements represent the bleed-through fraction relative to the primary channel.

    Parameters
    ----------
    wavelengths : npt.NDArray
        The common wavelength grid corresponding to the standardized spectra.
    standardized_spectra : List[npt.NDArray]
        List of 1D intensity arrays for each dye, corresponding to `wavelengths`.
        Expected length: n_dyes.
    collection_bands : List[Tuple[float, float]]
        List of (start_wavelength, stop_wavelength) tuples for each channel.
        Expected length: n_channels.

    Returns
    -------
    npt.NDArray
        The theoretical mixing matrix of shape (n_channels, n_dyes), with 1.0 on
        the diagonal.
    """
    n_dyes = len(standardized_spectra)
    n_channels = len(collection_bands)

    if n_dyes != n_channels:
        raise ValueError("Number of dyes must equal number of collection channels.")

    M = np.zeros((n_channels, n_dyes))

    for i, (band_start, band_stop) in enumerate(collection_bands):
        # Find indices corresponding to the collection band
        band_mask = (wavelengths >= band_start) & (wavelengths <= band_stop)

        for j, spectrum in enumerate(standardized_spectra):
            # Integrate (sum) the intensity over the band
            M[i, j] = np.sum(spectrum[band_mask])

    # Normalize each dye (column) by its "primary" channel (which we assume is channel j)
    # This ensures the diagonal is 1.0, and off-diagonal elements represent relative bleed-through
    for j in range(n_dyes):
        primary_intensity = M[j, j]
        if primary_intensity > 0:
            M[:, j] /= primary_intensity
        else:
            # If the dye doesn't emit at all in its primary channel, it's problematic
            # Leave as 0, but maybe warn?
            pass

    return M
