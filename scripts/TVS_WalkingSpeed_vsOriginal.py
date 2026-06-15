"""
TVS Full Pipeline Walking Speed Validation Figure
==================================================
Generates a 1x2 figure comparing walking speed estimation performance between
mobgap and the original Mobilise-D MATLAB implementation on the Mobilise-D
Technical Validation Study (TVS) free-living dataset.

Metrics shown:
- Panel A: Absolute Error (m/s)
- Panel B: Absolute Relative Error (%)

Significance annotations show Wilcoxon signed-rank test results (paired,
per-cohort) with Benjamini-Hochberg FDR correction applied across all tests.

Pre-computed validation results are fetched automatically from the mobgap
validation repository on GitHub (v1.2.0).

Usage
-----
    python tvs_full_pipeline_figures.py
    python tvs_full_pipeline_figures.py --output-dir ./results

Requirements
------------
    pip install mobgap matplotlib scipy statsmodels
"""

import argparse
from pathlib import Path
from typing import Optional
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mobgap.data.validation_results import ValidationResultLoader
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── Configuration ─────────────────────────────────────────────────────────────
__RESULT_VERSION = "v1.2.0"
COHORT_ORDER_PLOT = ["All", "HA", "CHF", "COPD", "MS", "PD", "PFF"]
COHORTS = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]
COLOURS = ["#555555", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

ALGORITHMS_MOBGAP_ONLY = {"Official_MobiliseD_Pipeline": "Official_MobiliseD_Pipeline"}
ALGORITHMS_BOTH = {
    "Official_MobiliseD_Pipeline": "mobgap",
    "EScience_MobiliseD_Pipeline": "MATLAB",
}
FREE_LIVING_INDEX_COLS = [
    "cohort", "participant_id", "time_measure",
    "recording", "recording_name", "recording_name_pretty",
]
LABORATORY_INDEX_COLS = [
    "cohort", "participant_id", "time_measure",
    "test", "trial", "test_name", "test_name_pretty",
]

# ── Data loading ──────────────────────────────────────────────────────────────
def format_loaded_results(
    values: dict[str, pd.DataFrame],
    index_cols: list[str],
    col_prefix_filter: Optional[str],
    convert_rel_error: bool = False,
) -> pd.DataFrame:
    """Reshape raw results dict into a clean flat DataFrame."""
    formatted = (
        pd.concat(values, names=["algo", *index_cols])
        .pipe(lambda df: df.filter(like=col_prefix_filter) if col_prefix_filter else df)
        .reset_index()
        .assign(_combined="combined")
    )
    if col_prefix_filter:
        formatted.columns = formatted.columns.str.removeprefix(col_prefix_filter)
    if convert_rel_error:
        rel_cols = [c for c in formatted.columns if "rel_error" in c]
        formatted[rel_cols] = formatted[rel_cols] * 100
    return formatted


def load_full_pipeline_both(version: str) -> pd.DataFrame:
    """Load full pipeline results for both mobgap and MATLAB."""
    loader = ValidationResultLoader("full_pipeline", result_path=None, version=version)
    parts = []
    for algo_name, source in ALGORITHMS_BOTH.items():
        fl = loader.load_single_results(algo_name, "free_living").reset_index()
        fl["source"] = source
        parts.append(fl)
    df = pd.concat(parts, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str)
    return df


# ── Statistical testing ───────────────────────────────────────────────────────
def compute_sig_results(df_fp: pd.DataFrame) -> dict:
    """
    Run paired Wilcoxon signed-rank tests between mobgap and MATLAB for
    walking speed absolute error and absolute relative error, per cohort.
    Applies Benjamini-Hochberg FDR correction across all tests.
    """
    sig_tests = [
        (df_fp, "matched__walking_speed_mps__abs_error",     "ws_abs"),
        (df_fp, "matched__walking_speed_mps__abs_rel_error", "ws_abrel"),
    ]

    sig_results = {}
    for df, col, label in sig_tests:
        for c in COHORTS:
            mob = df[(df["cohort"] == c) & (df["source"] == "mobgap")][["participant_id", col]]
            mat = df[(df["cohort"] == c) & (df["source"] == "MATLAB")][["participant_id", col]]
            merged = pd.merge(mob, mat, on="participant_id", suffixes=("_mob", "_mat")).dropna()
            if len(merged) > 0:
                sig_results[label, c] = stats.wilcoxon(
                    merged[f"{col}_mob"], merged[f"{col}_mat"]
                ).pvalue
            else:
                sig_results[label, c] = np.nan

    # Benjamini-Hochberg FDR correction
    sig_keys = list(sig_results.keys())
    p_values = [sig_results[k] for k in sig_keys]
    p_values_clean = [p if not np.isnan(p) else 1.0 for p in p_values]
    _, p_corrected, _, _ = multipletests(p_values_clean, method="fdr_bh")
    for key, p_corr, p_orig in zip(sig_keys, p_corrected, p_values):
        sig_results[key] = p_corr if not np.isnan(p_orig) else np.nan

    return sig_results


# ── Plotting helpers ──────────────────────────────────────────────────────────
def build_fp_paired(df: pd.DataFrame, col: str) -> tuple:
    """Build paired data lists for mobgap and MATLAB for boxplot."""
    def get_data(source):
        sub = df[df["source"] == source]
        all_vals = sub[col].dropna().values
        cohort_vals = [sub[sub["cohort"] == c][col].dropna().values for c in COHORTS]
        return [all_vals] + cohort_vals

    def get_ns(source):
        sub = df[df["source"] == source]
        return [len(sub)] + [len(sub[sub["cohort"] == c]) for c in COHORTS]

    return get_data("mobgap"), get_data("MATLAB"), get_ns("mobgap")


def get_sig_symbol(p: float) -> str:
    """Convert p-value to significance symbol."""
    if np.isnan(p):
        return ""
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def plot_clustered_panel_ws(
    ax,
    data_mobgap: list,
    data_matlab: list,
    ns: list,
    ylabel: str,
    panel_label: str,
    sig_results: dict,
    label: str,
) -> None:
    """Draw a single clustered walking speed box plot panel."""
    offset = 0.22
    positions_mobgap = [i - offset for i in range(1, 8)]
    positions_matlab  = [i + offset for i in range(1, 8)]
    width = 0.35

    for positions, data, source_label, hatch in [
        (positions_mobgap, data_mobgap, "mobgap", ""),
        (positions_matlab,  data_matlab, "MATLAB",  "///"),
    ]:
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(
                marker="o", markersize=3,
                markerfacecolor="none", markeredgewidth=0.6, alpha=0.4
            ),
            whiskerprops=dict(linewidth=1.1, color="#444444"),
            capprops=dict(linewidth=1.2, color="#444444"),
            zorder=2,
        )
        for patch, colour in zip(bp["boxes"], COLOURS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75 if source_label == "mobgap" else 0.35)
            patch.set_hatch(hatch)

    ax.axvline(1.5, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

    # Significance annotations
    y_top = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=y_top * 1.12)

    for i, c in enumerate(COHORTS, start=2):
        symbol = get_sig_symbol(sig_results.get((label, c), np.nan))
        if symbol:
            x_left  = positions_mobgap[i - 1]
            x_right = positions_matlab[i - 1]
            x_mid   = (x_left + x_right) / 2
            y_bracket = y_top * 1.02
            ax.plot([x_left, x_right], [y_bracket, y_bracket],
                    color="#333333", linewidth=1.0, clip_on=False)
            ax.plot([x_left,  x_left],  [y_bracket * 0.99, y_bracket],
                    color="#333333", linewidth=1.0, clip_on=False)
            ax.plot([x_right, x_right], [y_bracket * 0.99, y_bracket],
                    color="#333333", linewidth=1.0, clip_on=False)
            ax.text(x_mid, y_bracket * 1.005, symbol,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")

    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(
        [f"{c}\nn={n}" for c, n in zip(COHORT_ORDER_PLOT, ns)],
        fontsize=9
    )
    ax.set_xlim(0.3, 7.7)
    ax.set_xlabel("Cohort", fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.08, 1.08, panel_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data from GitHub validation repository...")
    df_fp = load_full_pipeline_both(__RESULT_VERSION)

    print("Running significance tests...")
    sig_results = compute_sig_results(df_fp)

    # Build plot data
    ws_abs_mobgap,   ws_abs_matlab,   ns_abs   = build_fp_paired(
        df_fp, "matched__walking_speed_mps__abs_error"
    )
    ws_abrel_mobgap, ws_abrel_matlab, ns_abrel = build_fp_paired(
        df_fp, "matched__walking_speed_mps__abs_rel_error"
    )

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97, top=0.84, bottom=0.18)

    plot_clustered_panel_ws(
        axes[0], ws_abs_mobgap, ws_abs_matlab,
        ns_abs, "Absolute Error (m/s)", "A", sig_results, "ws_abs"
    )
    plot_clustered_panel_ws(
        axes[1], ws_abrel_mobgap, ws_abrel_matlab,
        ns_abrel, "Absolute Relative Error (%)", "B", sig_results, "ws_abrel"
    )

    mobgap_patch = mpatches.Patch(facecolor="grey", alpha=0.75, label="mobgap")
    matlab_patch  = mpatches.Patch(
        facecolor="grey", alpha=0.35, hatch="///", label="Original Implementation"
    )
    fig.legend(
        handles=[mobgap_patch, matlab_patch],
        loc="lower center", ncol=2, frameon=True,
        framealpha=0.9, edgecolor="#cccccc",
        fontsize=10, bbox_to_anchor=(0.5, -0.02)
    )

    output_path = output_dir / "figure_walking_speed_vs_matlab.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TVS full pipeline walking speed comparison figure (mobgap vs MATLAB)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save output figures (default: current directory)",
    )
    args = parser.parse_args()
    main(args.output_dir)