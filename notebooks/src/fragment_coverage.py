"""Fragment coverage calculation and visualization.

This module re-exports from coverage.py and coverage_plot.py for backwards
compatibility. New code should import directly from those modules.
"""

from .coverage import (
    ALL_FRAGMENT_TYPES,
    C_TERM_IONS,
    N_TERM_IONS,
    calculate_coverage_from_labels,
    calculate_sequence_coverage,
    collect_fragment_assignments,
    parse_fragment_label,
)
from .coverage_plot import (
    DEFAULT_COLOR,
    ION_COLORS,
    calculate_coverage_with_slicer,
    format_ion_label_latex,
    format_legend_label,
    format_residue_labels,
    plot_fragment_coverage,
    plot_fragment_coverage_for_hash,
)

__all__ = [
    # Constants
    "ALL_FRAGMENT_TYPES",
    "C_TERM_IONS",
    "N_TERM_IONS",
    "ION_COLORS",
    "DEFAULT_COLOR",
    # Calculation
    "parse_fragment_label",
    "collect_fragment_assignments",
    "calculate_sequence_coverage",
    "calculate_coverage_from_labels",
    # Plotting
    "format_ion_label_latex",
    "format_legend_label",
    "format_residue_labels",
    "plot_fragment_coverage",
    "plot_fragment_coverage_for_hash",
    "calculate_coverage_with_slicer",
]
