"""
TVS Block-by-Block Validation Figure
=====================================
Generates a 2x2 figure comparing mobgap and the original Mobilise-D MATLAB
implementation across four algorithmic blocks (GSD, ICD, cadence, stride length)
on the Mobilise-D Technical Validation Study (TVS) free-living dataset.

Metrics shown:
- Gait Sequence Detection: F1 score
- Initial Contact Detection: F1 score
- Cadence Estimation: Absolute Relative Error (%)
- Stride Length Estimation: Absolute Relative Error (%)

Significance annotations show Wilcoxon signed-rank test results (paired,
per-cohort) with Benjamini-Hochberg FDR correction applied across all tests.

Pre-computed validation results are fetched automatically from the mobgap
validation repository on GitHub (v1.2.0).

Usage
-----
    python tvs_block_comparison_figures.py
    python tvs_block_comparison_figures.py --output-dir ./results

Requirements
------------
    pip install mobgap matplotlib scipy statsmodels
"""

import argparse
from pathlib import Path
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

# Algorithm mappings: algo_name -> (source_label, cohorts)
ALGORITHMS_FP = {
    "Official_MobiliseD_Pipeline": "mobgap",
    "EScience_MobiliseD_Pipeline": "MATLAB",
}
GSD_ALGO_MAP = {
    "GsdIluz":               ("mobgap", ["HA", "CHF", "COPD"]),
    "GsdIonescu":            ("mobgap", ["MS", "PD", "PFF"]),
    "matlab_TA_Iluz-original":   ("MATLAB", ["HA", "CHF", "COPD"]),
    "matlab_EPFL_V2-original":   ("MATLAB", ["MS", "PD", "PFF"]),
}
ICD_ALGO_MAP = {
    "IcdIonescu":          ("mobgap", None),
    "matlab_Ani_McCamley": ("MATLAB", None),
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_full_pipeline(version: str) -> pd.DataFrame:
    """Load full pipeline results for both mobgap and MATLAB."""
    loader = ValidationResultLoader("full_pipeline", result_path=None, version=version)
    parts = []
    for algo_name, source in ALGORITHMS_FP.items():
        fl = loader.load_single_results(algo_name, "free_living").reset_index()
        fl["source"] = source
        parts.append(fl)
    df = pd.concat(parts, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str)
    df.columns = [c.replace("matched__", "") if "matched__" in c else c for c in df.columns]
    # Convert relative errors from fraction to percentage
    df["cadence_spm__abs_rel_error"]     *= 100
    df["stride_length_m__abs_rel_error"] *= 100
    return df


def load_gsd(version: str) -> pd.DataFrame:
    """Load GSD results for both mobgap and MATLAB, split by cohort."""
    loader = ValidationResultLoader("gsd", result_path=None, version=version)
    parts = []
    for algo_name, (source, c_list) in GSD_ALGO_MAP.items():
        fl = loader.load_single_results(algo_name, "free_living").reset_index()
        fl["gs_absolute_relative_duration_error"] *= 100
        fl["source"] = source
        parts.append(fl[fl["cohort"].isin(c_list)])
    df = pd.concat(parts, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str)
    return df


def load_icd(version: str) -> pd.DataFrame:
    """Load ICD results for both mobgap and MATLAB."""
    loader = ValidationResultLoader("icd", result_path=None, version=version)
    parts = []
    for algo_name, (source, c_list) in ICD_ALGO_MAP.items():
        fl = loader.load_single_results(algo_name, "free_living").reset_index()
        fl["tp_relative_timing_error"] *= 100
        fl["source"] = source
        parts.append(fl)
    df = pd.concat(parts, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str)
    return df


# ── Statistical testing ───────────────────────────────────────────────────────
def compute_sig_results(
    df_gsd: pd.DataFrame,
    df_icd: pd.DataFrame,
    df_fp: pd.DataFrame,
) -> dict:
    """
    Run paired Wilcoxon signed-rank tests between mobgap and MATLAB for each
    metric and cohort, then apply Benjamini-Hochberg FDR correction.

    Returns a dict keyed by (metric_label, cohort) -> corrected p-value.
    """
    sig_tests = [
        (df_gsd, "f1_score",                     "gsd_f1"),
        (df_icd, "f1_score",                     "icd_f1"),
        (df_fp,  "cadence_spm__abs_rel_error",   "cadence"),
        (df_fp,  "stride_length_m__abs_rel_error", "stride_length"),
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

    # Apply Benjamini-Hochberg FDR correction across all tests
    sig_keys = list(sig_results.keys())
    p_values = [sig_results[k] for k in sig_keys]
    p_values_clean = [p if not np.isnan(p) else 1.0 for p in p_values]
    _, p_corrected, _, _ = multipletests(p_values_clean, method="fdr_bh")
    for key, p_corr, p_orig in zip(sig_keys, p_corrected, p_values):
        sig_results[key] = p_corr if not np.isnan(p_orig) else np.nan

    return sig_results


# ── Plotting helpers ──────────────────────────────────────────────────────────
def build_data_paired(df: pd.DataFrame, col: str) -> tuple:
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


def plot_clustered_panel(
    ax,
    data_mobgap: list,
    data_matlab: list,
    ns: list,
    ylabel: str,
    panel_label: str,
    sig_results: dict,
    label: str,
) -> None:
    """Draw a single clustered box plot panel with significance annotations."""
    offset = 0.22
    positions_mobgap = [i - offset for i in range(1, 8)]
    positions_matlab  = [i + offset for i in range(1, 8)]
    width = 0.35

    for positions, data, source_label, hatch in [
        (positions_mobgap, data_mobgap, "mobgap", ""),
        (positions_matlab,  data_matlab, "Original Implementation", "///"),
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
        fontsize=10
    )
    ax.set_xlim(0.3, 7.7)
    ax.set_xlabel("Cohort", fontsize=12, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=4)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.08, 1.06, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data from GitHub validation repository...")
    df_fp  = load_full_pipeline(__RESULT_VERSION)
    df_gsd = load_gsd(__RESULT_VERSION)
    df_icd = load_icd(__RESULT_VERSION)

    # Significance tests
    print("Running significance tests...")
    sig_results = compute_sig_results(df_gsd, df_icd, df_fp)

    # Build plot data
    gsd_mobgap, gsd_matlab, ns_gsd = build_data_paired(df_gsd, "f1_score")
    icd_mobgap, icd_matlab, ns_icd = build_data_paired(df_icd, "f1_score")
    cad_mobgap, cad_matlab, ns_cad = build_data_paired(df_fp,  "cadence_spm__abs_rel_error")
    sl_mobgap,  sl_matlab,  ns_sl  = build_data_paired(df_fp,  "stride_length_m__abs_rel_error")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), dpi=150)
    fig.subplots_adjust(hspace=0.45, wspace=0.28,
                        left=0.07, right=0.98, top=0.95, bottom=0.1)

    panels = [
        (gsd_mobgap, gsd_matlab, ns_gsd, "Gait Sequence Detection F1 Score",     "A", "gsd_f1"),
        (icd_mobgap, icd_matlab, ns_icd, "Initial Contact Detection F1 Score",    "B", "icd_f1"),
        (cad_mobgap, cad_matlab, ns_cad, "Cadence Abs. Relative Error (%)",       "C", "cadence"),
        (sl_mobgap,  sl_matlab,  ns_sl,  "Stride Length Abs. Relative Error (%)", "D", "stride_length"),
    ]

    for ax, (data_mobgap, data_matlab, ns, ylabel, panel_label, label) in zip(
        axes.flatten(), panels
    ):
        plot_clustered_panel(
            ax, data_mobgap, data_matlab, ns,
            ylabel, panel_label, sig_results, label
        )

    mobgap_patch = mpatches.Patch(facecolor="grey", alpha=0.75, label="mobgap")
    matlab_patch  = mpatches.Patch(
        facecolor="grey", alpha=0.35, hatch="///", label="Original Implementation"
    )
    fig.legend(
        handles=[mobgap_patch, matlab_patch],
        loc="lower center", ncol=2, frameon=True,
        framealpha=0.9, edgecolor="#cccccc",
        fontsize=11, bbox_to_anchor=(0.5, 0.0)
    )

    output_path = output_dir / "figure_block_vs_matlab.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TVS block-by-block comparison figure (mobgap vs MATLAB)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save output figures (default: current directory)",
    )
    args = parser.parse_args()
    main(args.output_dir)