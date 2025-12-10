import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from alphabase.peptide.precursor import hash_precursor_df
from alphabase.psm_reader import psm_reader_provider
from alphabase.psm_reader.dia_psm_reader import register_readers
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.tools.data_downloader import DataShareDownloader
from alphadia.raw_data.alpharaw_wrapper import Thermo
from peptdeep.pretrained_models import ModelManager

EXAMPLE_DIANN_REPORT_URL = (
    "https://datashare.biochem.mpg.de/s/Z8bOcBz6QViCVKN/download?"
)
EXAMPLE_DIANN_RAW_DATA_URL = (
    "https://datashare.biochem.mpg.de/s/ajTYDbBcVH0G6c2/download?"
)


def _download_diann_example_data(
    main_folder: Path, diann_report_url: str, raw_data_url: str
) -> tuple[Path, Path]:
    """Download DIA-NN data if not already present."""
    os.makedirs(main_folder, exist_ok=True)

    diann_report_path = DataShareDownloader(
        diann_report_url, str(main_folder)
    ).download()
    raw_file_path = DataShareDownloader(raw_data_url, str(main_folder)).download()

    return Path(diann_report_path), Path(raw_file_path)


def _coerce_raw_paths(
    raw_file_path: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    """Coerce raw file path(s) to a list of Path objects."""
    if raw_file_path is None:
        return []

    if isinstance(raw_file_path, (str, os.PathLike)):
        return [Path(raw_file_path)]

    return [Path(candidate) for candidate in raw_file_path]


def prepare_precursor_df(
    diann_df: pd.DataFrame, instrument: str, nce: float
) -> pd.DataFrame:
    precursor_df = diann_df.copy()
    precursor_df["instrument"] = instrument
    precursor_df["nce"] = nce
    precursor_df = hash_precursor_df(precursor_df)
    precursor_df["rt"] = precursor_df["rt"] * 60
    precursor_df["rt_start"] = precursor_df["rt_start"] * 60
    precursor_df["rt_stop"] = precursor_df["rt_stop"] * 60

    return precursor_df


def prepare_speclib_base(
    precursor_df: pd.DataFrame,
    charged_frag_types: list[str],
    instrument: str,
    nce: float,
    model_device: str,
) -> SpecLibBase:
    model_mgr = ModelManager(device=model_device)
    model_mgr.instrument = instrument
    model_mgr.nce = nce

    res = model_mgr.predict_all(precursor_df.copy())

    speclib_base = SpecLibBase(charged_frag_types=charged_frag_types)
    speclib_base._precursor_df = res["precursor_df"]
    speclib_base._fragment_mz_df = res["fragment_mz_df"]
    speclib_base._fragment_intensity_df = res["fragment_intensity_df"]

    return speclib_base


def load_diann_data(
    diann_report_path: str | Path | None = None,
    raw_file_path: str | Path | Sequence[str | Path] | None = None,
    *,
    main_folder: Path | None = None,
    download_urls: tuple[str, str] | None = None,
    model_device: str = "cpu",
    charged_frag_types: list[str] | None = None,
) -> tuple[pd.DataFrame, SpecLibBase, SpecLibFlat, Any]:
    """Load DIA-NN data and create spectral libraries with automatic instrument/NCE detection.

    Args:
        diann_report_path: Path to DIA-NN report file (e.g., report.parquet)
        raw_file_path: Path (or list of paths) to .raw files
        main_folder: Folder for downloading data (required if download_urls provided)
        download_urls: Optional tuple (diann_report_url, raw_file_url) to download data
        model_device: Device for model computation ("cpu" or "gpu")
        charged_frag_types: Fragment ion types to include

    Returns:
        Tuple of (precursor_df, speclib_base, spectral_library_flat, dia_data)
    """
    if charged_frag_types is None:
        charged_frag_types = [
            "b_z1",
            "b_z2",
            "y_z1",
            "y_z2",
            "b_modloss_z1",
            "b_modloss_z2",
            "y_modloss_z1",
            "y_modloss_z2",
        ]

    raw_path_input: str | Path | Sequence[str | Path] | None = None
    if download_urls:
        if main_folder is None:
            raise ValueError("main_folder must be provided when using download_urls")
        diann_report_url, raw_file_url = download_urls
        diann_report_path, raw_file_path = _download_diann_example_data(
            main_folder, diann_report_url, raw_file_url
        )
        raw_path_input = raw_file_path
    else:
        raw_path_input = raw_file_path

    raw_paths = _coerce_raw_paths(raw_path_input)

    if diann_report_path is None or not raw_paths:
        raise ValueError(
            "Both diann_report_path and raw_file_path must be provided (either directly or via download_urls)"
        )
    instrument = "Orbitrap"

    dia_data_map: dict[str, Thermo] = {}
    nce_by_run: dict[str, float] = {}
    for raw_path in raw_paths:
        if raw_path.suffix.lower() != ".raw":
            raise ValueError(
                f"Unsupported raw format: {raw_path.suffix}. Expecting .raw"
            )
        dia_data = Thermo(str(raw_path))
        raw_name = raw_path.stem
        dia_data_map[raw_name] = dia_data
        nce_value = 28.0
        if "nce" in dia_data.spectrum_df.columns and not dia_data.spectrum_df.empty:
            nce_value = float(dia_data.spectrum_df["nce"].max())
        nce_by_run[raw_name] = nce_value

    default_nce = max(nce_by_run.values())

    register_readers()
    modification_mapping = {
        "Phospho@S": "S(Phospho)",
        "Phospho@T": "T(Phospho)",
        "Phospho@Y": "Y(Phospho)",
    }
    diann_reader = psm_reader_provider.get_reader(
        "diann", modification_mapping=modification_mapping
    )
    diann_df = diann_reader.import_file(str(diann_report_path))
    diann_df = diann_df[diann_df["raw_name"].isin(dia_data_map.keys())]
    if diann_df.empty:
        raise ValueError(
            "No DIA-NN entries matched the provided RAW files. "
            "Ensure the 'Run' column matches the RAW filenames."
        )

    precursor_df = prepare_precursor_df(diann_df, instrument, default_nce)
    speclib_base = prepare_speclib_base(
        precursor_df, charged_frag_types, instrument, default_nce, model_device
    )

    spectral_library_flat = SpecLibFlat()
    spectral_library_flat.parse_base_library(speclib_base)

    precursor_df["mods"] = precursor_df["mods"].replace("", np.nan)
    precursor_df["mod_sites"] = precursor_df["mod_sites"].replace("", np.nan)

    dia_data_result: Any
    if len(dia_data_map) == 1:
        dia_data_result = next(iter(dia_data_map.values()))
    else:
        dia_data_result = dia_data_map

    return precursor_df, speclib_base, spectral_library_flat, dia_data_result
