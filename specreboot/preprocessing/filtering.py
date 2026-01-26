from typing import List, Tuple

from matchms import Spectrum
from matchms.filtering import normalize_intensities  # optional
from matchms.filtering.default_pipelines import DEFAULT_FILTERS, CLEAN_PEAKS
from matchms.filtering.SpectrumProcessor import SpectrumProcessor


def general_cleaning(
    spectra: List[Spectrum],
    file_name: str,
    create_report: bool = True
) -> Tuple[List[Spectrum], dict]:
    """
    Clean and normalize spectra using the default Matchms filters.

    See:
    https://github.com/matchms/matchms/blob/33704801aa19b31ddeb1c636271236fdcb4b70d9
    /matchms/filtering/default_pipelines.py#L77

    Parameters
    ----------
    spectra : list of Spectrum
        Input spectra to be cleaned.
    file_name : str
        Name for the output cleaned spectra file (used by Matchms).
    create_report : bool, optional
        Whether to generate a Matchms filter report, by default True.

    Returns
    -------
    cleaned_spectra : list of Spectrum
        The processed spectra after applying Matchms filters.
    report : dict
        Processing report returned by the SpectrumProcessor.
    """
    spectrum_processor = SpectrumProcessor(DEFAULT_FILTERS + CLEAN_PEAKS)

    cleaned_spectra, report = spectrum_processor.process_spectra(
        spectra,
        cleaned_spectra_file=file_name,
        create_report=create_report,
    )

    return cleaned_spectra, report

