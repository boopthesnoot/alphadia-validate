"""Mirror plot visualization for comparing observed vs theoretical spectra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair
import numpy as np
import pandas as pd
import seaborn as sns

from .xic_utils import correlation_coefficient, median_axis, normalize_profiles

if TYPE_CHECKING:
    from .slicer import SpectrumSlicer

FRAGMENT_COLORS = {
    "a": "#388E3C",
    "b": "#1976D2",
    "c": "#00796B",
    "x": "#7B1FA2",
    "y": "#D32F2F",
    "z": "#F57C00",
    "unknown": "#212121",
    None: "#808080",
}


def _get_colors(n_colors, palette_name=None):
    """Get a color palette for RT scans."""
    if palette_name is not None:
        pal = sns.color_palette(palette_name, n_colors)
        return list(pal.as_hex())

    pal = sns.color_palette("Spectral", n_colors + 3)
    colors = list(pal.as_hex())
    # Remove middle elements to avoid washed-out colors
    if len(colors) >= 6:
        mid = len(colors) // 2
        del colors[mid - 1 : mid + 2]
    return colors[:n_colors]


def split_dense_by_rt(dense, mz_library, label_library):
    """Split dense spectrum array by retention time."""
    intensity_observed_full = dense[0].sum(axis=(1, 2))

    reconstructed_indices = np.unravel_index(
        np.arange(intensity_observed_full.size), intensity_observed_full.shape
    )
    mz_flat = mz_library[reconstructed_indices[0]]
    label_flat = label_library[reconstructed_indices[0]]
    intensity_flat = intensity_observed_full.ravel()

    indices = np.indices(intensity_observed_full.shape)
    rt_indices = indices[1].ravel()

    df_obs = pd.DataFrame(
        {
            "mz": mz_flat,
            "frag_label": label_flat,
            "intensity": intensity_flat,
            "rt_idx": rt_indices,
        }
    )
    df_obs["rt_cat"] = pd.Categorical(df_obs.rt_idx)
    max_int = df_obs.groupby("mz")["intensity"].sum().max()
    df_obs["norm_intensity"] = df_obs["intensity"] / max_int
    df_obs["frag_label"] = df_obs.apply(
        lambda x: "" if x.intensity <= 0 else x.frag_label, axis=1
    )

    return df_obs


def convert_library_to_df(mz_library, intensity_library):
    """Convert library arrays to DataFrame."""
    df_library = pd.DataFrame({"mz": mz_library, "intensity": intensity_library})
    df_library["norm_intensity"] = df_library["intensity"] / df_library["intensity"].max()
    return df_library


def format_precursor_entry(prec):
    """Format precursor entry for display."""
    if isinstance(prec.mods, str):
        seq = list(prec.sequence)
        mods = [m.split("@")[0] for m in prec.mods.split(";")]
        sites = np.array(prec.mod_sites.split(";"), dtype=int) - 1
        for i, s in enumerate(sites):
            seq[s] = f"{seq[s]}[{mods[i]}]"
        seq = "".join(seq)
    else:
        seq = prec.sequence
    return f"{seq} (+{prec.charge})"


def add_corr(df_obs, df_library):
    """Add correlation coefficients to observation DataFrame."""
    df_merged = df_library.merge(df_obs, on="mz")
    df_merged.columns = [
        c.replace("_x", "_pred").replace("_y", "_obs") for c in df_merged.columns
    ]

    df_merged = df_merged.query("intensity_pred!=0 & intensity_obs!=0")
    df_corr = (
        df_merged.groupby("rt_idx")[["intensity_pred", "intensity_obs"]]
        .corr()
        .reset_index()
        .melt(id_vars=["rt_idx", "level_1"])
        .query("value!=1")
        .query('level_1=="intensity_obs"')
        .rename(columns={"value": "corr_coeff"})[["corr_coeff", "rt_idx"]]
    )
    df_obs_corr = df_obs.merge(df_corr, on="rt_idx")

    df_obs_corr["rt_cat"] = df_obs_corr.apply(
        lambda x: f"{x.rt_idx}, r={round(x.corr_coeff, 2)}", axis=1
    )
    return df_obs_corr


def plot_mirror_by_rt(
    dense,
    mz_library,
    intensity_library,
    label_library,
    precursor_entry,
    add_corr_coeff=True,
    width=800,
    height=300,
):
    """Create mirror plot comparing observed vs theoretical spectra by RT."""
    seq = format_precursor_entry(precursor_entry)
    df_library = convert_library_to_df(mz_library, intensity_library)
    df_obs = split_dense_by_rt(dense, mz_library, label_library)

    if add_corr_coeff:
        df_obs = add_corr(df_obs, df_library)

    return plot_mirror_by_rt_from_dfs(df_obs, df_library, seq, width, height)


def plot_mirror_by_rt_from_dfs(df_obs, df_library, title="", width=800, height=300):
    """Create mirror plot from DataFrames."""
    theo = _plot_theo(df_library)
    obs = _plot_obs_by_rt(df_obs)

    middle_line = (
        altair.Chart(pd.DataFrame({"sep": [0]}))
        .mark_rule(size=3)
        .encode(y="sep", color=altair.value("lightGray"))
    )

    title_params = altair.TitleParams(text=title, fontWeight="bold", color="black")

    return (obs + theo + middle_line).properties(
        width=width, height=height, title=title_params
    )


def _plot_obs_by_rt(df_plot):
    """Plot observed spectrum colored by RT scan."""
    annotation_kws = {"align": "left", "angle": 270, "baseline": "middle"}

    anno = [
        altair.Tooltip("mz", format=".3f", title="m/z"),
        altair.Tooltip("norm_intensity", format=".2f", title="Intensity"),
    ]

    df_plot = df_plot.copy()
    df_plot["label_color"] = df_plot["frag_label"].apply(
        lambda x: FRAGMENT_COLORS.get(x[0] if x else None, "#0000FF")
    )

    df_plot["rt_idx"] = df_plot["rt_cat"].apply(
        lambda x: int(x.split(",")[0]) if isinstance(x, str) and "," in x else x
    )

    rt_categories = df_plot.sort_values("rt_idx")["rt_cat"].unique().tolist()
    rt_colors = _get_colors(n_colors=len(rt_categories))

    color = altair.Color(
        "rt_cat",
        scale=altair.Scale(domain=rt_categories, range=rt_colors),
        title="RT Scan",
    )
    x = altair.X(
        "mz",
        axis=altair.Axis(title="m/z", titleFontStyle="italic", grid=True),
        scale=altair.Scale(
            nice=False,
            padding=5,
            zero=False,
            domain=[50, (max(df_plot["mz"]) // 10) * 10 + 50],
        ),
    )
    y = altair.Y(
        "sum(norm_intensity):Q",
        axis=altair.Axis(title="Intensity", format=".2f", grid=True),
    )
    color2 = altair.Color("label_color", scale=None)

    frag_anno = (
        altair.Chart(df_plot)
        .mark_text(dx=10, **annotation_kws)
        .encode(x=x, y=y, text="frag_label", color=color2)
    )

    return frag_anno + (
        altair.Chart(df_plot)
        .mark_bar(size=3)
        .encode(x=x, y=y, color=color, tooltip=anno)
    )


def _plot_theo(df_plot):
    """Plot theoretical spectrum (inverted)."""
    anno = [
        altair.Tooltip("mz", format=".3f", title="m/z"),
        altair.Tooltip("intensity", format=".2f", title="Intensity"),
    ]

    df_plot = df_plot.copy()
    df_plot["minus_intensity"] = -df_plot["norm_intensity"]

    x = altair.X(
        "mz",
        axis=altair.Axis(title="m/z", titleFontStyle="italic", grid=True),
        scale=altair.Scale(
            nice=False,
            padding=5,
            zero=False,
            domain=[50, (max(df_plot["mz"]) // 10) * 10 + 50],
        ),
    )
    y = altair.Y(
        "minus_intensity",
        axis=altair.Axis(title="Intensity", format="", grid=True),
        scale=altair.Scale(nice=True, padding=0),
    )
    return altair.Chart(df_plot).mark_rule(size=3).encode(x=x, y=y, tooltip=anno)


def plot_xic_with_background(spectrum_slice, palette_name=None, hex_colors=None):
    """Plot XIC with colored background for each RT scan."""
    xic_observed = spectrum_slice[0].sum(axis=(1, 2))
    df_xic = (
        pd.DataFrame(xic_observed).reset_index().rename(columns={"index": "mz_scan"})
    )
    df_xic = df_xic.melt(
        id_vars=["mz_scan"], var_name="RT_scan", value_name="intensity"
    )
    n_colors = df_xic["RT_scan"].nunique()

    median_data = df_xic.groupby("RT_scan", as_index=False)["intensity"].mean()

    if hex_colors is not None:
        background_colors = hex_colors[:n_colors]
    else:
        background_colors = _get_colors(palette_name=palette_name, n_colors=n_colors)

    background = (
        altair.Chart(df_xic)
        .mark_bar(opacity=0.05)
        .encode(
            x=altair.X("RT_scan:O", axis=altair.Axis(title="RT scan")),
            y=altair.value(1),
            color=altair.Color(
                "RT_scan:O",
                scale=altair.Scale(
                    domain=list(df_xic["RT_scan"].unique()), range=background_colors
                ),
                legend=None,
            ),
        )
        .properties(width=400, height=300)
    )

    base_chart = (
        altair.Chart(df_xic)
        .mark_line()
        .encode(
            x=altair.X(
                "RT_scan:Q", axis=altair.Axis(title=None, labels=False, ticks=False)
            ),
            y="intensity:Q",
            color=altair.Color(
                "mz_scan:Q",
                scale=altair.Scale(scheme="greys"),
                legend=altair.Legend(title="Fragment No."),
            ),
        )
    )

    median_chart = (
        altair.Chart(median_data)
        .mark_line(color="red")
        .encode(
            x=altair.X(
                "RT_scan:Q", axis=altair.Axis(title=None, labels=False, ticks=False)
            ),
            y="intensity:Q",
        )
    )

    return background + base_chart + median_chart


def plot_correlations(spectrum_slice, mz_library):
    """Plot correlation coefficients for each fragment."""
    intensity_slice = spectrum_slice[0].sum(axis=1).sum(axis=1)
    normalized_intensity_slice = normalize_profiles(intensity_slice)
    median_profile = median_axis(normalized_intensity_slice, axis=0)
    corr_list = correlation_coefficient(median_profile, intensity_slice)

    df_corrs = pd.DataFrame({"mz": mz_library, "corr_coeff": corr_list})

    return (
        altair.Chart(df_corrs)
        .mark_circle()
        .encode(
            x=altair.X("mz:Q", axis=altair.Axis(title="m/z")),
            y=altair.X("corr_coeff:Q", axis=altair.Axis(title="Correlation")),
            size=altair.Size(
                "corr_coeff:Q",
                legend=altair.Legend(title="Correlation"),
            ),
        )
    )


def mirror_with_xic_and_corrs(
    spectrum_slice: np.ndarray,
    mz_library: np.ndarray,
    intensity_library: np.ndarray,
    fragment_library: np.ndarray,
    precursor_entry: pd.Series,
    *,
    width: int = 600,
    height: int = 300,
):
    """Create combined mirror plot with XIC and correlations.

    Low-level function that takes pre-fetched data.

    Args:
        spectrum_slice: Observed spectrum data from slicer.get_by_hash().
        mz_library: Fragment m/z values from slicer.get_by_hash().
        intensity_library: Theoretical intensities from slicer.get_by_hash().
        fragment_library: Fragment labels from slicer.get_by_hash().
        precursor_entry: Single precursor row (pd.Series) with sequence, mods, charge.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        Altair chart with mirror plot, XIC, and correlations.
    """
    mirror = plot_mirror_by_rt(
        spectrum_slice,
        mz_library,
        intensity_library,
        fragment_library,
        precursor_entry,
        width=width * 0.8,
        height=height,
    )
    xic = plot_xic_with_background(spectrum_slice)
    corrs = plot_correlations(spectrum_slice, mz_library)

    mirror_corrs = altair.vconcat(
        mirror.properties(width=width * 0.75, height=height * 0.8),
        corrs.properties(width=width * 0.75, height=height * 0.2),
    ).resolve_scale(x="shared", color="independent", size="independent")

    return (
        (mirror_corrs | xic.properties(width=width * 0.25, height=height * 0.5))
        .resolve_scale(color="independent")
        .configure_view(stroke=None)
    )


def mirror_with_xic_and_corrs_for_hash(
    slicer: SpectrumSlicer,
    selected_hash: int,
    *,
    raw_name: str | None = None,
    width: int = 600,
    height: int = 300,
):
    """Create combined mirror plot with XIC and correlations for a precursor hash.

    High-level convenience function that fetches data from a slicer.

    Args:
        slicer: Initialized SpectrumSlicer instance.
        selected_hash: The mod_seq_charge_hash to plot.
        raw_name: RAW file name (required if hash appears in multiple files).
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        Altair chart with mirror plot, XIC, and correlations.
    """
    mz_library, intensity_library, spectrum_slice, fragment_library = slicer.get_by_hash(
        selected_hash, raw_name=raw_name
    )
    precursor_entry = slicer._select_precursor_entry(selected_hash, raw_name)

    return mirror_with_xic_and_corrs(
        spectrum_slice,
        mz_library,
        intensity_library,
        fragment_library,
        precursor_entry,
        width=width,
        height=height,
    )
