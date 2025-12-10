"""Fragment coverage calculation for peptide sequences."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# N-terminal ion series (forward direction in alphabase)
N_TERM_IONS = frozenset({"a", "b", "c", "b_modloss", "b_H2O", "b_NH3", "c_lossH"})
# C-terminal ion series (reverse direction in alphabase)
C_TERM_IONS = frozenset({"x", "y", "z", "y_modloss", "y_H2O", "y_NH3", "z_addH"})
# All supported fragment types (from alphabase)
ALL_FRAGMENT_TYPES = N_TERM_IONS | C_TERM_IONS

# Regex pattern for fragment labels: {ion}{number}{_loss}(+{charge})
# Examples: b3(+1), y5_H2O(+1), b4_NH3(+2)
_FRAGMENT_LABEL_PATTERN = re.compile(
    r"^([abcxyz])(\d+)(?:_(\w+))?\(\+(\d+)\)$"
)


def parse_fragment_label(label: str) -> tuple[str, int, int] | None:
    """Parse a fragment label into (ion_type, ordinal, charge).

    Args:
        label: Fragment label in format '{ion}{number}{_loss}(+{charge})'
            Examples: 'b3(+1)', 'y5_H2O(+1)', 'b4_NH3(+2)'

    Returns:
        Tuple of (ion_type, ordinal, charge) or None if parsing fails.
        ion_type includes loss suffix if present (e.g., 'b', 'y_H2O', 'b_NH3').
    """
    match = _FRAGMENT_LABEL_PATTERN.match(label)
    if not match:
        return None

    base_ion, ordinal_str, loss_type, charge_str = match.groups()

    # Build full ion type
    if loss_type:
        # Map 'modification_loss' to 'modloss' for consistency with alphabase
        if loss_type == "modification_loss":
            loss_type = "modloss"
        ion_type = f"{base_ion}_{loss_type}"
    else:
        ion_type = base_ion

    return ion_type, int(ordinal_str), int(charge_str)


def collect_fragment_assignments(
    fragment_labels: Sequence[str] | NDArray[np.str_],
    observed_intensity: np.ndarray | None,
    theoretical_intensity: np.ndarray | None,
    allowed_series: set[str],
    threshold: float = 0.0,
) -> dict[tuple[str, int], dict]:
    """Select a representative fragment per ion/ordinal pair.

    For each unique (ion_type, ordinal) combination, keeps the fragment with
    the highest intensity score. Observed intensities take precedence over
    theoretical predictions.

    Args:
        fragment_labels: Fragment labels from spectral library.
        observed_intensity: Experimental intensities (same length as labels).
        theoretical_intensity: Predicted intensities (same length as labels).
        allowed_series: Set of ion types to include (e.g., {'b', 'y', 'b_H2O'}).
        threshold: Minimum intensity to consider a fragment.

    Returns:
        Dict mapping (ion_type, ordinal) to fragment info dict with keys:
        'ion_type', 'ordinal', 'charge', 'status', 'score'.
    """
    assignments: dict[tuple[str, int], dict] = {}

    for idx, raw_label in enumerate(fragment_labels):
        parsed = parse_fragment_label(str(raw_label))
        if parsed is None:
            continue

        ion_type, ordinal, charge = parsed
        if ion_type not in allowed_series:
            continue

        key = (ion_type, ordinal)

        # Get intensity values
        obs_val = observed_intensity[idx] if observed_intensity is not None else np.nan
        theo_val = (
            theoretical_intensity[idx] if theoretical_intensity is not None else np.nan
        )

        # Determine status and score
        if not np.isnan(obs_val) and obs_val > threshold:
            score = float(obs_val)
            status = "observed"
        elif not np.isnan(theo_val) and theo_val > threshold:
            score = float(theo_val)
            status = "predicted"
        else:
            continue

        # Keep highest scoring fragment for each position
        current = assignments.get(key)
        if current is None or score > current["score"]:
            assignments[key] = {
                "ion_type": ion_type,
                "ordinal": ordinal,
                "charge": charge,
                "status": status,
                "score": score,
            }

    return assignments


def calculate_sequence_coverage(
    sequence: str,
    assignments: dict[tuple[str, int], dict],
) -> float:
    """Calculate percentage of peptide bonds covered by observed fragments.

    A peptide bond is considered covered if any observed fragment ion
    (N-terminal or C-terminal) corresponds to cleavage at that position.

    Args:
        sequence: Peptide amino acid sequence.
        assignments: Fragment assignments from collect_fragment_assignments().

    Returns:
        Coverage as float between 0.0 and 1.0.
    """
    if len(sequence) <= 1:
        return 0.0

    n_bonds = len(sequence) - 1
    observed_bonds: set[int] = set()

    for (ion_type, ordinal), entry in assignments.items():
        if entry["status"] != "observed":
            continue

        if ion_type in N_TERM_IONS:
            # b3 covers bond 3 (between 3rd and 4th AA)
            bond_idx = ordinal
        elif ion_type in C_TERM_IONS:
            # y3 covers bond N-3
            bond_idx = len(sequence) - ordinal
        else:
            continue

        if 1 <= bond_idx <= n_bonds:
            observed_bonds.add(bond_idx)

    return len(observed_bonds) / n_bonds


def calculate_coverage_from_labels(
    sequence: str,
    fragment_labels: Sequence[str] | NDArray[np.str_],
    observed_intensity: np.ndarray | None = None,
    theoretical_intensity: np.ndarray | None = None,
    fragment_types: Iterable[str] | None = None,
) -> float:
    """Calculate sequence coverage from fragment labels and intensities.

    Convenience function that combines collect_fragment_assignments and
    calculate_sequence_coverage.

    Args:
        sequence: Peptide amino acid sequence.
        fragment_labels: Fragment labels from spectral library.
        observed_intensity: Experimental intensities.
        theoretical_intensity: Predicted intensities.
        fragment_types: Ion types to consider. Defaults to ALL_FRAGMENT_TYPES.

    Returns:
        Coverage as float between 0.0 and 1.0.
    """
    allowed = set(fragment_types) if fragment_types else ALL_FRAGMENT_TYPES

    assignments = collect_fragment_assignments(
        fragment_labels, observed_intensity, theoretical_intensity, allowed
    )

    return calculate_sequence_coverage(sequence, assignments)
