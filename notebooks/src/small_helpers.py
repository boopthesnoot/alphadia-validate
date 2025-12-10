"""Small helper utilities for precursor selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PrecursorInfo:
    """Information about a selected precursor."""

    index: int
    hash: int
    sequence: str
    qval: float


def get_random_precursor_hash(
    precursor_df: pd.DataFrame,
    quantile: int,
    *,
    n_quantiles: int = 100,
) -> int:
    """Get a random precursor hash from a specific q-value quantile.

    Args:
        precursor_df: DataFrame with 'qval' and 'mod_seq_charge_hash' columns.
        quantile: Quantile index (0 = best q-values, 99 = worst).
        n_quantiles: Number of quantiles to split into (default 100).

    Returns:
        mod_seq_charge_hash of a randomly selected precursor.
    """
    precursor_df_sorted = precursor_df.sort_values("qval")
    quantiles = np.array_split(precursor_df_sorted, n_quantiles)
    random_precursor = quantiles[quantile].sample(n=1)

    return int(random_precursor["mod_seq_charge_hash"].values[0])


def get_random_precursor_info(
    precursor_df: pd.DataFrame,
    quantile: int,
    *,
    n_quantiles: int = 100,
) -> PrecursorInfo:
    """Get info about a random precursor from a specific q-value quantile.

    Args:
        precursor_df: DataFrame with 'qval', 'sequence', and 'mod_seq_charge_hash' columns.
        quantile: Quantile index (0 = best q-values, 99 = worst).
        n_quantiles: Number of quantiles to split into (default 100).

    Returns:
        PrecursorInfo with index, hash, sequence, and q-value.
    """
    precursor_df_sorted = precursor_df.sort_values("qval")
    quantiles = np.array_split(precursor_df_sorted, n_quantiles)
    random_precursor = quantiles[quantile].sample(n=1)

    return PrecursorInfo(
        index=int(random_precursor.index.values[0]),
        hash=int(random_precursor["mod_seq_charge_hash"].values[0]),
        sequence=str(random_precursor["sequence"].values[0]),
        qval=float(random_precursor["qval"].values[0]),
    )
