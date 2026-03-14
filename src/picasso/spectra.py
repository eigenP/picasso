import json
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


def interpolate_spectrum(
    wavelengths: npt.NDArray,
    intensities: npt.NDArray,
    new_wavelengths: npt.NDArray,
) -> npt.NDArray:
    """
    Interpolate a spectrum to a new set of wavelengths.

    Any values outside the original wavelength range are set to 0.

    Parameters
    ----------
    wavelengths : npt.NDArray
        Original wavelengths.
    intensities : npt.NDArray
        Original intensities.
    new_wavelengths : npt.NDArray
        New wavelengths to interpolate to.

    Returns
    -------
    npt.NDArray
        Interpolated intensities at `new_wavelengths`.
    """
    f = interp1d(
        wavelengths,
        intensities,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return f(new_wavelengths)


def load_fpbase_spectra(
    file_path: str,
) -> List[Tuple[npt.NDArray, npt.NDArray]]:
    """
    Load spectra from an FPbase JSON export.

    The FPbase API or export typically provides a list of spectra objects,
    each with `data` containing `[wavelength, intensity]` pairs.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing FPbase spectra.

    Returns
    -------
    List[Tuple[npt.NDArray, npt.NDArray]]
        A list of (wavelengths, intensities) arrays.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Depending on the exact FPbase format, data might be a list or a dict containing a list
    # Assuming data is a list of objects with a 'data' or 'spectrum' key.
    spectra_list = []

    # Handle the structure where the file contains a list of dye objects
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "spectra" in data:
            items = data["spectra"]
        else:
            # Maybe it's a single spectrum
            items = [data]
    else:
        raise ValueError("Unsupported JSON format for FPbase spectra.")

    for item in items:
        # Some endpoints return "data" (list of [x, y] pairs), others "spectrum"
        spec_data = item.get("data") or item.get("spectrum")

        if not spec_data:
            raise ValueError("Could not find spectrum data in the JSON object.")

        spec_array = np.array(spec_data)

        if spec_array.shape[1] != 2:
            raise ValueError(f"Expected [x, y] pairs, got shape {spec_array.shape}")

        wavelengths = spec_array[:, 0]
        intensities = spec_array[:, 1]

        spectra_list.append((wavelengths, intensities))

    return spectra_list


def standardize_spectra(
    spectra: List[Tuple[npt.NDArray, npt.NDArray]],
    start_wl: float = 300.0,
    end_wl: float = 900.0,
    step: float = 1.0,
) -> Tuple[npt.NDArray, List[npt.NDArray]]:
    """
    Standardize a list of spectra to a common wavelength grid.

    Parameters
    ----------
    spectra : List[Tuple[npt.NDArray, npt.NDArray]]
        List of (wavelengths, intensities) for each spectrum.
    start_wl : float, optional
        Start wavelength, by default 300.0
    end_wl : float, optional
        End wavelength, by default 900.0
    step : float, optional
        Wavelength step size, by default 1.0

    Returns
    -------
    Tuple[npt.NDArray, List[npt.NDArray]]
        The common wavelength array and the list of interpolated intensities.
    """
    new_wavelengths = np.arange(start_wl, end_wl + step, step)

    standardized = []
    for wl, intensities in spectra:
        interp_intensities = interpolate_spectrum(wl, intensities, new_wavelengths)
        # normalize to peak = 1.0
        max_int = interp_intensities.max()
        if max_int > 0:
            interp_intensities /= max_int
        standardized.append(interp_intensities)

    return new_wavelengths, standardized
