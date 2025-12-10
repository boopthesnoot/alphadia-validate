"""Isoform attention plotting for phosphorylation site localization."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alphabase.protein.fasta import SpecLibFasta
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.translate import create_modified_sequence

from .slicer import SpectrumSlicer


def get_isoform_id(
    mods: str,
    mod_sites: str,
    translate_mod_dict: dict | None = None,
    ignore_mods: set | None = None,
) -> str:
    """Convert modifications and sites into an isoform identifier string.

    Args:
        mods: Semicolon-separated modification names (e.g., 'Carbamidomethyl@C;Oxidation@M').
        mod_sites: Semicolon-separated 1-based site positions (e.g., '7;24').
        translate_mod_dict: Optional mapping of modification names for display.
        ignore_mods: Modifications to exclude from isoform ID (e.g., fixed mods).

    Returns:
        Isoform identifier string (e.g., 'Phospho5;Phospho12') or 'Base' if no variable mods.
    """
    if ignore_mods is None:
        ignore_mods = set()

    if not mods:
        return "Base"

    var_mods = []
    var_mod_sites = []
    for mod, site in zip(mods.split(";"), mod_sites.split(";")):
        if mod not in ignore_mods:
            var_mods.append(mod)
            var_mod_sites.append(int(site))

    if not var_mods:
        return "Base"

    # Sort by site position
    order = np.argsort(var_mod_sites)
    var_mod_sites = [var_mod_sites[i] for i in order]
    var_mods = [var_mods[i] for i in order]

    # Format modification names
    if translate_mod_dict is None:
        var_mods = [mod[: mod.find("@")] for mod in var_mods]
    else:
        var_mods = [translate_mod_dict[mod] for mod in var_mods]

    return ";".join(f"{mod}{site}" for site, mod in zip(var_mod_sites, var_mods))


def fragment_isoform_annotation(
    precursor_entry: pd.Series,
    fixed_mods: set | None = None,
) -> pd.DataFrame:
    """Generate fragment table with isoform annotations for all possible localizations.

    Args:
        precursor_entry: Precursor entry from precursor_df.
        fixed_mods: Set of fixed modifications to ignore for isoform generation.

    Returns:
        DataFrame with fragment m/z values and isoform annotations.
    """
    if fixed_mods is None:
        fixed_mods = {"Carbamidomethyl@C"}

    sequence = precursor_entry["sequence"]
    charge = precursor_entry["charge"]

    mods_str = precursor_entry.get("mods", "")
    if pd.isna(mods_str):
        mods_str = ""

    mod_counter = Counter(mods_str.split(";")) if mods_str else Counter()

    # Count variable modifications
    var_mod_counts = [
        mod_counter[mod] for mod in mod_counter if mod not in fixed_mods
    ]
    max_var_mod_num = max(var_mod_counts) if var_mod_counts else 0

    potential_fragments = [
        f"{ion_type}_z{z}" for ion_type in ["b", "y"] for z in range(1, charge + 1)
    ]

    precursor_df = pd.DataFrame({"sequence": [sequence], "charge": [charge]})

    fastalib = SpecLibFasta(
        potential_fragments,
        var_mods=[mod for mod in mod_counter if mod not in fixed_mods],
        fix_mods=[mod for mod in mod_counter if mod in fixed_mods],
        min_var_mod_num=max_var_mod_num,
        max_var_mod_num=max_var_mod_num,
    )

    fastalib.precursor_df = precursor_df
    fastalib.add_modifications()
    fastalib.calc_fragment_mz_df()

    flatlib = SpecLibFlat()
    flatlib.parse_base_library(fastalib)

    fragment_table = flatlib.fragment_df.copy()
    fragment_table["isoform"] = "Base"

    for idx in range(len(flatlib.precursor_df)):
        entry = flatlib.precursor_df.iloc[idx]
        if entry["mod_sites"]:
            start = entry["flat_frag_start_idx"]
            stop = entry["flat_frag_stop_idx"]
            fragment_table.loc[start:stop, "isoform"] = get_isoform_id(
                entry["mods"], entry["mod_sites"], ignore_mods=fixed_mods
            )

    return fragment_table


def find_isoform(mz: float, isoform_map: pd.DataFrame, match_tolerance: float) -> str:
    """Find matching isoform(s) for a given m/z value."""
    matching = isoform_map.loc[
        np.abs(isoform_map["mz"] - mz) / mz < match_tolerance, "isoform"
    ]
    return "+".join(matching.unique())


def isoform_attention_plot(
    spectral_library_flat: SpecLibFlat,
    precursor_df: pd.DataFrame,
    dia_data,
    selected_hash: int,
    match_tolerance: float = 7e-6,
    raw_name: str | None = None,
):
    """Create isoform attention plot showing which peaks support which localization.

    Args:
        spectral_library_flat: Flat spectral library from alphabase.
        precursor_df: Precursor DataFrame with mod_seq_charge_hash column.
        dia_data: DIA data object(s) for retrieving spectra.
        selected_hash: The mod_seq_charge_hash to plot.
        match_tolerance: Relative m/z tolerance for matching (default 7 ppm).
        raw_name: RAW file name (required if hash appears in multiple files).
    """
    slicer = SpectrumSlicer(spectral_library_flat, precursor_df, dia_data)

    precursor_entry = slicer._select_precursor_entry(selected_hash, raw_name)
    precursor_entry = precursor_entry.fillna("")

    print(precursor_entry[["sequence", "mods", "mod_sites"]])

    mz_library, intensity_library, data_slice, _ = slicer.get_by_hash(
        selected_hash, raw_name=raw_name
    )

    isoform_map = fragment_isoform_annotation(precursor_entry)

    library_spec = pd.DataFrame({"mz": mz_library, "intensity": intensity_library})
    library_spec["isoform"] = library_spec["mz"].apply(
        find_isoform, args=(isoform_map, match_tolerance)
    )

    intensity_observed = data_slice[0].sum(axis=(1, 2, 3))
    intensity_observed_normalized = intensity_observed / intensity_observed.max()
    library_spec["intensity"] = library_spec["intensity"] / library_spec["intensity"].max()

    plt.stem(
        mz_library,
        intensity_observed_normalized,
        markerfmt="None",
        basefmt="grey",
        label="Observed",
    )

    for n, (isoform_id, peak_group) in enumerate(library_spec.groupby("isoform")):
        plt.stem(
            peak_group["mz"],
            -1 * peak_group["intensity"],
            basefmt="grey",
            linefmt=f"C{n + 1}-",
            label=isoform_id,
            markerfmt="None",
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(
        create_modified_sequence(
            (
                precursor_entry["sequence"],
                precursor_entry["mods"],
                precursor_entry["mod_sites"],
            ),
            nterm="",
            cterm="",
        )
    )
    plt.xlabel("m/z")
    plt.ylabel("Relative Intensity")
    plt.show()
