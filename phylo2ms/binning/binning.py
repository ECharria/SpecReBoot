import numpy as np
from matchms import Spectrum
from typing import List


def compute_global_bins(spectra: List[Spectrum], decimals: int) -> List[float]:
    """
    Compute global m/z bins across all spectra by rounding peak m/z values.

    Parameters
    ----------
    spectra : list of Spectrum
        Collection of spectra to extract and bin m/z values from.
    decimals : int
        Number of decimal places to round m/z values to.

    Returns
    -------
    list of float
        Sorted list of unique rounded m/z bin values.
    """
    all_binned_mz = [
        mz
        for spec in spectra
        for mz in np.round(spec.peaks.mz, decimals)
    ]

    return sorted(set(all_binned_mz))


def bin_spectra(spectra: List[Spectrum], decimals: int) -> List[Spectrum]:
    """
    Create new spectra with rounded m/z values (binned peaks).

    Parameters
    ----------
    spectra : list of Spectrum
        List of input spectra.
    decimals : int
        Number of decimal places to round m/z values to.

    Returns
    -------
    list of Spectrum
        New spectra with binned/rounded m/z values.
    """
    binned_spectra = []

    for spec in spectra:
        mz_rounded = np.round(spec.peaks.mz, decimals)
        new_spec = Spectrum(
            mz=mz_rounded,
            intensities=spec.peaks.intensities.copy(),
            metadata=spec.metadata.copy() if spec.metadata else None,
        )
        binned_spectra.append(new_spec)

    return binned_spectra
