"""I/O utilities for loading AlphaDIA data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.tools.data_downloader import DataShareDownloader
from alphadia.data.alpharaw_wrapper import MzML, Sciex, Thermo
from alphadia.data.bruker import TimsTOFTranspose

EXAMPLE_RAW_DATA_URL = "https://datashare.biochem.mpg.de/s/VfqtW5p9MJ0kxAC/download?files=20231017_OA2_TiHe_ADIAMA_HeLa_200ng_Evo011_21min_F-40_07.mzML"
EXAMPLE_PRECURSORS_TSV_URL = (
    "https://datashare.biochem.mpg.de/s/VfqtW5p9MJ0kxAC/download?files=precursors.tsv"
)
EXAMPLE_SPECLIB_URL = (
    "https://datashare.biochem.mpg.de/s/VfqtW5p9MJ0kxAC/download?files=speclib.hdf"
)


def load_alphadia_data(
    main_folder: Path,
    *,
    download_urls: tuple[str, str, str] | None = None,
    raw_file_name: str | None = None,
    precursors_file_name: str | None = None,
    speclib_file_name: str | None = None,
) -> tuple[pd.DataFrame, SpecLibBase, SpecLibFlat, Any]:
    """Load AlphaDIA results (precursors.tsv + speclib.hdf) and raw data.

    Args:
        main_folder: Folder containing data or download destination.
        download_urls: Optional tuple (raw_url, precursors_url, speclib_url) for downloading.
        raw_file_name: Name of the raw file (required if not downloading).
        precursors_file_name: Name of the precursors file (required if not downloading).
        speclib_file_name: Name of the speclib file (required if not downloading).

    Returns:
        Tuple of (precursor_df, spectral_library, spectral_library_flat, dia_data).
    """
    if download_urls:
        precursors_tsv_path, raw_file_path, speclib_path = (
            _download_alphadia_example_data(main_folder, *download_urls)
        )
    else:
        if not all([precursors_file_name, raw_file_name, speclib_file_name]):
            raise ValueError(
                "Provide file names for precursors, raw file, and speclib"
            )

        precursors_tsv_path = main_folder / precursors_file_name
        raw_file_path = main_folder / raw_file_name
        speclib_path = main_folder / speclib_file_name

    current_raw_name = raw_file_path.stem
    precursor_df = pd.read_csv(precursors_tsv_path, sep="\t")
    precursor_df = precursor_df[precursor_df["raw_name"] == current_raw_name]

    spectral_library = SpecLibBase()
    spectral_library.load_hdf(speclib_path)

    print("Reading raw file ...")
    dia_data = _load_raw_file(raw_file_path)

    print("Parsing spectral library ...")
    spectral_library_flat = SpecLibFlat()
    spectral_library_flat.parse_base_library(spectral_library)

    return precursor_df, spectral_library, spectral_library_flat, dia_data


def _load_raw_file(raw_file_path: Path) -> Any:
    """Load raw data file based on extension."""
    suffix = raw_file_path.suffix.lower()
    loaders = {
        ".mzml": MzML,
        ".raw": Thermo,
        ".wiff": Sciex,
        ".d": TimsTOFTranspose,
    }

    if suffix not in loaders:
        supported = ", ".join(loaders.keys())
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {supported}")

    return loaders[suffix](str(raw_file_path))


def _download_alphadia_example_data(
    main_folder: Path, raw_data_url: str, precursors_tsv_url: str, speclib_url: str
) -> tuple[Path, Path, Path]:
    """Download AlphaDIA data if not already present."""
    os.makedirs(main_folder, exist_ok=True)

    raw_file_path = DataShareDownloader(raw_data_url, str(main_folder)).download()
    precursors_tsv_path = DataShareDownloader(
        precursors_tsv_url, str(main_folder)
    ).download()
    speclib_path = DataShareDownloader(speclib_url, str(main_folder)).download()

    return Path(precursors_tsv_path), Path(raw_file_path), Path(speclib_path)


def get_spectral_library_summary(spectral_library: SpecLibBase) -> dict[str, pd.DataFrame]:
    """Get summary DataFrames from a spectral library.

    Args:
        spectral_library: SpecLibBase instance.

    Returns:
        Dict with 'precursor_df', 'fragment_mz_df', 'fragment_intensity_df',
        and 'flat_fragment_df' DataFrames.
    """
    spectral_library_flat = SpecLibFlat()
    spectral_library_flat.parse_base_library(spectral_library)

    return {
        "precursor_df": spectral_library.precursor_df[
            [
                "precursor_mz",
                "sequence",
                "mods",
                "mod_sites",
                "charge",
                "mod_seq_charge_hash",
                "frag_start_idx",
                "frag_stop_idx",
            ]
        ],
        "fragment_mz_df": spectral_library.fragment_mz_df,
        "fragment_intensity_df": spectral_library.fragment_intensity_df,
        "flat_fragment_df": spectral_library_flat.fragment_df,
    }
