"""Spectrum slicing utilities for DIA data visualization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat

# Ion type mappings from alphabase (ASCII codes)
_TYPE_MAP = {97: "a", 98: "b", 99: "c", 120: "x", 121: "y", 122: "z"}

# Loss type mappings from alphabase LOSS_MAPPING
_LOSS_MAP = {
    0: "",
    1: "lossH",
    2: "addH",
    17: "NH3",
    18: "H2O",
    98: "modification_loss",
}


def _make_fragment_label(row: pd.Series) -> str:
    """Create fragment label from alphabase fragment_df row."""
    ion = _TYPE_MAP[row["type"]]
    loss = _LOSS_MAP[row["loss_type"]]
    loss_suffix = f"_{loss}" if loss else ""
    return f"{ion}{int(row['number'])}{loss_suffix}(+{int(row['charge'])})"


class SpectrumSlicer:
    """Extract and visualize spectrum slices for precursors from DIA data."""

    def __init__(
        self,
        spectral_library_flat: SpecLibFlat,
        precursor_df: pd.DataFrame,
        dia_data: Any | Mapping[str, Any],
    ):
        """Initialize the slicer.

        Args:
            spectral_library_flat: Flat spectral library from alphabase.
            precursor_df: DataFrame with precursor information including
                mod_seq_charge_hash and frame/scan or RT indices.
            dia_data: Single DIA data object or mapping of raw_name -> DIA data
                for multi-file analysis.
        """
        self._add_fragment_labels(spectral_library_flat)
        self.spectral_library_flat = spectral_library_flat
        self.precursor_df = precursor_df
        self._jit_data_map = self._build_jit_data_map(dia_data)
        self._add_missing_indices()

    def _add_fragment_labels(self, spectral_library_flat: SpecLibFlat) -> None:
        """Add fragment_label column to fragment_df if missing."""
        if "fragment_label" not in spectral_library_flat.fragment_df.columns:
            spectral_library_flat.fragment_df["fragment_label"] = (
                spectral_library_flat.fragment_df.apply(_make_fragment_label, axis=1)
            )

    def get_by_hash(
        self, selected_hash: int, raw_name: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get spectrum slice for a precursor hash.

        Args:
            selected_hash: The mod_seq_charge_hash to query.
            raw_name: RAW file name (required if hash appears in multiple files).

        Returns:
            Tuple of (mz_library, intensity_library, spectrum_slice, fragment_labels).
            spectrum_slice is a 5D array: [info_type, fragment, mobility, cycle, rt].
        """
        speclib_entry, mz_library, intensity_library, fragment_labels = (
            self._get_library_entry(selected_hash)
        )
        precursor_entry = self._select_precursor_entry(selected_hash, raw_name)

        precursor_query = np.array(
            [[speclib_entry.precursor_mz, speclib_entry.precursor_mz]], dtype=np.float32
        )
        scan_limits = np.array(
            [[precursor_entry.scan_start, precursor_entry.scan_stop, 1]], dtype=np.int64
        )
        frame_limits = np.array(
            [[precursor_entry.frame_start, precursor_entry.frame_stop, 1]], dtype=np.int64
        )

        jit_data = self._get_jit_for_raw_name(precursor_entry.get("raw_name"))
        spectrum_slice, _ = jit_data.get_dense(
            frame_limits, scan_limits, mz_library, 30, precursor_query
        )

        return mz_library, intensity_library, spectrum_slice, fragment_labels

    def _get_library_entry(
        self, hash_: int, min_intensity: float = 0.01
    ) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]:
        """Get spectral library entry for a hash."""
        speclib_entry = self.spectral_library_flat.precursor_df[
            self.spectral_library_flat.precursor_df["mod_seq_charge_hash"] == hash_
        ].iloc[0]

        start = speclib_entry.flat_frag_start_idx
        stop = speclib_entry.flat_frag_stop_idx
        frag_df = self.spectral_library_flat.fragment_df.iloc[start:stop]

        mz = frag_df["mz"].to_numpy()
        intensity = frag_df["intensity"].to_numpy()
        labels = frag_df["fragment_label"].to_numpy()

        # Filter by intensity and sort by m/z
        mask = intensity > min_intensity
        order = np.argsort(mz[mask])

        return (
            speclib_entry,
            mz[mask][order],
            intensity[mask][order],
            labels[mask][order],
        )

    def _select_precursor_entry(
        self, selected_hash: int, raw_name: str | None
    ) -> pd.Series:
        """Select precursor entry, optionally filtering by raw_name."""
        rows = self.precursor_df[
            self.precursor_df["mod_seq_charge_hash"] == selected_hash
        ]
        if rows.empty:
            raise KeyError(f"No precursor found for hash {selected_hash}")

        normalized = _normalize_raw_name(raw_name)
        if normalized is not None:
            if "raw_name" not in rows.columns:
                raise ValueError("raw_name filtering requested but column missing")
            rows = rows[rows["raw_name"].astype(str) == normalized]
            if rows.empty:
                raise KeyError(f"No precursor for hash {selected_hash} in {normalized}")
        elif "raw_name" in rows.columns and rows["raw_name"].nunique(dropna=False) > 1:
            raise ValueError(
                f"Hash {selected_hash} appears in multiple RAW files. Specify raw_name."
            )

        return rows.iloc[0]

    def _build_jit_data_map(self, dia_data: Any | Mapping[str, Any]) -> dict[str, Any]:
        """Build mapping of raw_name -> jit data object."""
        if isinstance(dia_data, Mapping):
            return {str(k): v.to_jitclass() for k, v in dia_data.items()}

        # Single dia_data object
        raw_names = self._available_raw_names()
        if len(raw_names) > 1:
            raise ValueError(
                "Multiple raw files in precursor_df but single dia_data provided. "
                "Pass a mapping of raw_name -> dia_data."
            )

        name = next(iter(raw_names), "raw_0")
        return {name: dia_data.to_jitclass()}

    def _available_raw_names(self) -> set[str]:
        """Get set of raw names from precursor_df."""
        if "raw_name" not in self.precursor_df.columns:
            return set()
        return set(self.precursor_df["raw_name"].dropna().astype(str).unique())

    def _get_jit_for_raw_name(self, raw_name: Any) -> Any:
        """Get jit data for a raw name."""
        normalized = _normalize_raw_name(raw_name)
        if normalized is None:
            if len(self._jit_data_map) == 1:
                return next(iter(self._jit_data_map.values()))
            raise ValueError("Multiple RAW files present. Specify raw_name.")

        if normalized not in self._jit_data_map:
            available = ", ".join(self._jit_data_map.keys())
            raise KeyError(f"Unknown raw_name '{normalized}'. Available: {available}")

        return self._jit_data_map[normalized]

    def _add_missing_indices(self) -> None:
        """Add frame/scan indices from RT values if missing."""
        required = ["frame_start", "frame_stop", "scan_start", "scan_stop"]
        present = [c for c in required if c in self.precursor_df.columns]

        if len(present) == len(required):
            return
        if present:
            raise ValueError(f"Partial DIA indices present: {present}. Need all or none.")

        if "rt_start" not in self.precursor_df.columns:
            raise ValueError("Cannot calculate indices: rt_start/rt_stop missing")

        print("Converting RT values to DIA cycle frame indices...")

        frame_start = pd.Series(index=self.precursor_df.index, dtype=np.int64)
        frame_stop = pd.Series(index=self.precursor_df.index, dtype=np.int64)

        if "raw_name" in self.precursor_df.columns:
            groups = list(self.precursor_df.groupby("raw_name", dropna=False))
        else:
            groups = [(None, self.precursor_df)]

        for raw_name, group in groups:
            jit_data = self._get_jit_for_raw_name(raw_name)

            for idx, row in group.iterrows():
                rt_limits = np.array([row["rt_start"], row["rt_stop"]], dtype=np.float32)
                frames = jit_data._get_frame_indices(rt_limits, 1, 1)
                frame_start.loc[idx] = frames[0, 0]
                frame_stop.loc[idx] = frames[0, 1]

        self.precursor_df["frame_start"] = frame_start
        self.precursor_df["frame_stop"] = frame_stop
        self.precursor_df["scan_start"] = 0
        self.precursor_df["scan_stop"] = 0


def _normalize_raw_name(raw_name: Any) -> str | None:
    """Normalize raw_name to string or None."""
    if raw_name is None or pd.isna(raw_name):
        return None
    return str(raw_name)
