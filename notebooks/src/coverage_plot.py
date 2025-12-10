"""Fragment coverage visualization for peptide sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .coverage import (
    ALL_FRAGMENT_TYPES,
    C_TERM_IONS,
    N_TERM_IONS,
    calculate_sequence_coverage,
    collect_fragment_assignments,
)
from .slicer import SpectrumSlicer

# Ion colors - loss variants use same color as base ion
ION_COLORS: Mapping[str, str] = {
    "a": "#388E3C",
    "b": "#1976D2",
    "c": "#00796B",
    "x": "#7B1FA2",
    "y": "#D32F2F",
    "z": "#F57C00",
    "b_modloss": "#1976D2",
    "b_H2O": "#1976D2",
    "b_NH3": "#1976D2",
    "c_lossH": "#00796B",
    "y_modloss": "#D32F2F",
    "y_H2O": "#D32F2F",
    "y_NH3": "#D32F2F",
    "z_addH": "#F57C00",
}
DEFAULT_COLOR = "#212121"

# Loss type display formatting
_LOSS_DISPLAY_LATEX = {
    "H2O": r"H_2O",
    "NH3": r"NH_3",
    "modloss": "mod",
    "lossH": "H",
    "addH": "+H",
}

_LOSS_DISPLAY_UNICODE = {
    "H2O": "H₂O",
    "NH3": "NH₃",
    "modloss": "mod",
    "lossH": "H",
    "addH": "+H",
}


def format_ion_label_latex(ion_type: str, ordinal: int) -> str:
    """Format ion type and ordinal as a LaTeX label.

    Examples:
        ('b', 3) -> '$b_{3}$'
        ('y_H2O', 5) -> '$y_{5}^{-H_2O}$'
        ('z_addH', 4) -> '$z_{4}^{+H}$'
    """
    parts = ion_type.split("_", 1)
    base_ion = parts[0]

    if len(parts) == 1:
        return rf"${base_ion}_{{{ordinal}}}$"

    loss_type = parts[1]
    loss_display = _LOSS_DISPLAY_LATEX.get(loss_type, loss_type)

    # addH is an addition, not a loss
    if loss_type == "addH":
        return rf"${base_ion}_{{{ordinal}}}^{{{loss_display}}}$"
    return rf"${base_ion}_{{{ordinal}}}^{{-{loss_display}}}$"


def format_legend_label(ion_type: str) -> str:
    """Format ion type for legend display (e.g., 'b-ions', 'y-H₂O-ions')."""
    parts = ion_type.split("_", 1)
    if len(parts) == 1:
        return f"{ion_type}-ions"

    base, loss = parts
    loss_display = _LOSS_DISPLAY_UNICODE.get(loss, loss)
    return f"{base}-{loss_display}-ions"


def format_residue_labels(
    sequence: str,
    mods: str | None,
    mod_sites: str | None,
) -> list[str]:
    """Create per-residue text labels including modification annotations.

    Args:
        sequence: Peptide amino acid sequence.
        mods: Semicolon-separated modification names (e.g., 'Oxidation@M;Phospho@S').
        mod_sites: Semicolon-separated 1-based site positions (e.g., '5;12').

    Returns:
        List of labels, one per residue, with modifications appended.
    """
    labels = list(sequence)

    # Handle missing or NaN values
    if mods is None or mod_sites is None:
        return labels
    if bool(pd.isna(mods)) or bool(pd.isna(mod_sites)):
        return labels
    if not mods or not mod_sites:
        return labels

    mod_list = [m for m in str(mods).split(";") if m]
    site_list = [s for s in str(mod_sites).split(";") if s]

    if len(mod_list) != len(site_list):
        return labels

    prefix_parts: list[str] = []
    suffix_parts: list[str] = []

    for mod, site_str in zip(mod_list, site_list):
        # Skip non-numeric sites
        if not site_str.lstrip("-").isdigit():
            continue

        site = int(site_str)
        clean_mod = mod.split("@")[0]
        tag = f"[{clean_mod}]"

        if site > 0:
            idx = min(site - 1, len(labels) - 1)
            labels[idx] = f"{labels[idx]}{tag}"
        elif site == 0:
            prefix_parts.append(tag)
        else:
            suffix_parts.append(tag)

    if prefix_parts:
        labels[0] = "".join(prefix_parts) + labels[0]
    if suffix_parts:
        labels[-1] = labels[-1] + "".join(suffix_parts)

    return labels


def _build_cleavage_map(
    assignments: dict[tuple[str, int], dict],
    pep_len: int,
) -> dict[int, dict[str, dict | None]]:
    """Build map of cleavage positions to N-term/C-term fragment entries."""
    cleavage_map: dict[int, dict[str, dict | None]] = {}

    for (ion_type, ordinal), entry in assignments.items():
        if ion_type in N_TERM_IONS:
            position = ordinal
            direction = "n_term"
        elif ion_type in C_TERM_IONS:
            position = pep_len - ordinal
            direction = "c_term"
        else:
            continue

        if not (0 < position < pep_len):
            continue

        slot = cleavage_map.setdefault(position, {"n_term": None, "c_term": None})
        current = slot[direction]
        if current is None or entry["score"] > current["score"]:
            slot[direction] = entry

    return cleavage_map


def _draw_fragment_annotations(
    ax: Axes,
    cleavage_map: dict[int, dict[str, dict | None]],
    residue_boundaries: np.ndarray,
    spacing_height: float,
    tick_length: float,
) -> set[str]:
    """Draw fragment ion annotations and return set of observed ion types."""
    legend_series: set[str] = set()

    for position, entries in sorted(cleavage_map.items()):
        if position >= len(residue_boundaries):
            continue

        x = float(residue_boundaries[position])

        # C-terminal (top, pointing right)
        c_entry = entries["c_term"]
        if c_entry:
            ion_type = c_entry["ion_type"]
            legend_series.add(ion_type)
            color = ION_COLORS.get(ion_type, DEFAULT_COLOR)

            ax.add_line(Line2D([x, x], [0.0, spacing_height], color=color, lw=2, zorder=2))
            ax.add_line(Line2D([x, x + tick_length], [spacing_height] * 2, color=color, lw=2, zorder=2))
            ax.text(
                x + tick_length + 0.05, spacing_height,
                format_ion_label_latex(ion_type, c_entry["ordinal"]),
                ha="left", va="bottom", fontsize=11, fontweight="bold", color=color,
            )

        # N-terminal (bottom, pointing left)
        n_entry = entries["n_term"]
        if n_entry:
            ion_type = n_entry["ion_type"]
            legend_series.add(ion_type)
            color = ION_COLORS.get(ion_type, DEFAULT_COLOR)

            ax.add_line(Line2D([x, x], [0.0, -spacing_height], color=color, lw=2, zorder=2))
            ax.add_line(Line2D([x, x - tick_length], [-spacing_height] * 2, color=color, lw=2, zorder=2))
            ax.text(
                x - tick_length - 0.05, -spacing_height,
                format_ion_label_latex(ion_type, n_entry["ordinal"]),
                ha="right", va="top", fontsize=11, fontweight="bold", color=color,
            )

    return legend_series


def plot_fragment_coverage(
    sequence: str,
    fragment_labels: Sequence[str] | NDArray[np.str_],
    *,
    observed_intensity: np.ndarray | None = None,
    theoretical_intensity: np.ndarray | None = None,
    fragment_types: Iterable[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    residue_labels: Sequence[str] | None = None,
) -> Axes:
    """Draw fragment-ion coverage zigzag over a peptide sequence.

    N-terminal ions (a, b, c and variants) are shown below the sequence.
    C-terminal ions (x, y, z and variants) are shown above.

    Args:
        sequence: Peptide amino acid sequence.
        fragment_labels: Fragment labels from spectral library.
        observed_intensity: Experimental intensities aligned with labels.
        theoretical_intensity: Predicted intensities aligned with labels.
        fragment_types: Ion series to include. Defaults to all supported types.
        ax: Optional matplotlib Axes to draw on.
        title: Optional plot title (coverage % is always appended).
        residue_labels: Optional custom residue labels (e.g., with modifications).

    Returns:
        Matplotlib Axes with the coverage plot.
    """
    allowed_series = set(fragment_types) if fragment_types else set(ALL_FRAGMENT_TYPES)

    assignments = collect_fragment_assignments(
        fragment_labels, observed_intensity, theoretical_intensity, allowed_series
    )

    if not assignments:
        raise ValueError("No fragment ions matched the requested fragment_types.")

    coverage = calculate_sequence_coverage(sequence, assignments)
    pep_len = len(sequence)

    # Calculate layout dimensions
    display_labels = list(residue_labels) if residue_labels else list(sequence)
    char_width = 0.35
    residue_widths = [1.0 + char_width * max(0, len(lbl) - 5) for lbl in display_labels]
    boundaries = np.concatenate(([0.0], np.cumsum(residue_widths)))
    centers = boundaries[:-1] + np.array(residue_widths) / 2.0
    total_width = boundaries[-1]

    max_label_len = max(len(lbl) for lbl in display_labels)
    spacing = 1.1 + 0.12 * max(0, max_label_len - 1)
    tick_len = min(0.7, 0.45 * float(np.median(residue_widths)))

    # Create axes if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, total_width * 0.6), 4))

    # Build cleavage map and draw annotations
    cleavage_map = _build_cleavage_map(assignments, pep_len)
    legend_series = _draw_fragment_annotations(ax, cleavage_map, boundaries, spacing, tick_len)

    # Draw residue labels
    for pos, label in zip(centers, display_labels):
        ax.text(pos, 0, label, ha="center", va="center", fontsize=12, fontweight="bold", family="monospace", zorder=3)

    # Configure axes
    ax.set_xlim(0, total_width)
    ax.set_ylim(-spacing - 1.0, spacing + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title with coverage
    title_text = f"{title} (Coverage: {coverage:.1%})" if title else f"Coverage: {coverage:.1%}"
    ax.set_title(title_text, fontsize=14, pad=16)

    # Legend
    if legend_series:
        handles = [
            Line2D([0], [0], color=ION_COLORS.get(s, DEFAULT_COLOR), lw=2, label=format_legend_label(s))
            for s in sorted(legend_series)
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)

    return ax


def plot_fragment_coverage_for_hash(
    slicer: SpectrumSlicer,
    selected_hash: int,
    *,
    raw_name: str | None = None,
    fragment_types: Iterable[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot fragment coverage for a precursor hash using a SpectrumSlicer.

    Args:
        slicer: Initialized SpectrumSlicer instance.
        selected_hash: The mod_seq_charge_hash to plot.
        raw_name: RAW file name (required if hash appears in multiple files).
        fragment_types: Ion series to include. Defaults to all supported types.
        ax: Optional matplotlib Axes to draw on.
        title: Optional plot title.

    Returns:
        Matplotlib Axes with the coverage plot.
    """
    _, intensity_library, spectrum_slice, fragment_labels = slicer.get_by_hash(
        selected_hash, raw_name=raw_name
    )
    observed_intensity = spectrum_slice[0].sum(axis=(1, 2, 3))

    precursor_entry = slicer._select_precursor_entry(selected_hash, raw_name)
    sequence = str(precursor_entry["sequence"])

    residue_labels = format_residue_labels(
        sequence,
        precursor_entry.get("mods"),
        precursor_entry.get("mod_sites"),
    )

    return plot_fragment_coverage(
        sequence,
        fragment_labels,
        observed_intensity=observed_intensity,
        theoretical_intensity=intensity_library,
        fragment_types=fragment_types,
        ax=ax,
        title=title,
        residue_labels=residue_labels,
    )


def calculate_coverage_with_slicer(
    slicer: SpectrumSlicer,
    selected_hash: int,
    *,
    raw_name: str | None = None,
    fragment_types: Iterable[str] | None = None,
) -> float:
    """Calculate sequence coverage for a precursor hash using an existing slicer.

    Args:
        slicer: Initialized SpectrumSlicer instance.
        selected_hash: The mod_seq_charge_hash to analyze.
        raw_name: RAW file name (required if hash appears in multiple files).
        fragment_types: Ion series to consider. Defaults to all supported types.

    Returns:
        Coverage as float between 0.0 and 1.0.
    """
    _, intensity_library, spectrum_slice, fragment_labels = slicer.get_by_hash(
        selected_hash, raw_name=raw_name
    )
    observed_intensity = spectrum_slice[0].sum(axis=(1, 2, 3))

    precursor_entry = slicer._select_precursor_entry(selected_hash, raw_name)
    sequence = str(precursor_entry["sequence"])

    allowed = set(fragment_types) if fragment_types else set(ALL_FRAGMENT_TYPES)
    assignments = collect_fragment_assignments(
        fragment_labels, observed_intensity, intensity_library, allowed
    )

    return calculate_sequence_coverage(sequence, assignments)
