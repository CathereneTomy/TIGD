"""
chembl_activity_utils.py
========================
Utility functions for ChEMBL activity data processing pipeline.

Usage (notebook):
    from chembl_activity_utils import (
        load_and_clean,
        get_scaffold,
        passes_threshold,
        process_standard_type,
        compute_summary_counts,
        save_summary_report,
        plot_publications_per_year,
        run_pipeline,
    )
"""

import warnings
warnings.filterwarnings("ignore")

import os
from io import StringIO
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ── Potency rules & thresholds ────────────────────────────────────────────────

POTENCY_RULES = {
    "IC50":       True,
    "EC50":       True,
    "Ki":         True,
    "Kd":         True,
    "XC50":       True,
    "AC50":       True,
    "ED50":       True,
    "LD50":       False,
    "Inhibition": False,
    "Activity":   False,
}

POTENCY_THRESHOLDS = {
    "IC50":       {"max": 10_000, "min": None},
    "EC50":       {"max": 10_000, "min": None},
    "Ki":         {"max":  1_000, "min": None},
    "Kd":         {"max":  1_000, "min": None},
    "XC50":       {"max": 10_000, "min": None},
    "AC50":       {"max": 10_000, "min": None},
    "ED50":       {"max": 10_000, "min": None},
    "Inhibition": {"max": None,   "min": 50},
    "Activity":   {"max": None,   "min": 50},
    "LD50":       {"max": None,   "min": 1_000},
}

SELECTED_COLUMNS = [
    "Molecule ChEMBL ID", "Molecule Name", "Molecule Max Phase",
    "Molecular Weight", "Smiles", "Standard Type", "Standard Value",
    "Standard Units", "Data Validity Comment", "Assay ChEMBL ID",
    "Assay Description", "Assay Type", "BAO Label", "Assay Organism",
    "Document ChEMBL ID", "Document Year",
]


# ── Chemistry helpers ─────────────────────────────────────────────────────────

def get_scaffold(smiles: str) -> str | None:
    """Return the Murcko scaffold SMILES for a molecule, or None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


def passes_threshold(
    row: pd.Series,
    stype: str,
    thresholds: dict = POTENCY_THRESHOLDS,
) -> bool:
    """Return True if the row's Standard Value passes the potency threshold."""
    val = row["Standard Value"]
    thresh = thresholds.get(stype, {})
    if thresh.get("max") is not None and val > thresh["max"]:
        return False
    if thresh.get("min") is not None and val < thresh["min"]:
        return False
    return True


# ── Step 1: Load & clean ──────────────────────────────────────────────────────

def load_and_clean(
    file_path: str,
    selected_columns: list[str] = SELECTED_COLUMNS,
) -> tuple[pd.DataFrame, int]:
    """
    Load a ChEMBL CSV file (semicolon-separated, commas in values) and
    return a cleaned DataFrame plus the pre-threshold unique molecule count.

    Parameters
    ----------
    file_path        : Path to the raw ChEMBL CSV.
    selected_columns : Columns to keep (defaults to SELECTED_COLUMNS).

    Returns
    -------
    df_clean               : Cleaned DataFrame.
    total_unique_molecules : Row count before any threshold is applied.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().replace(",", " ")

    df = pd.read_csv(StringIO(content), sep=";")
    df_clean = df[selected_columns].copy()

    df_clean = df_clean.dropna(subset=["Standard Value", "Smiles"])
    df_clean = df_clean[
        (df_clean["Standard Value"].astype(str).str.strip() != "") &
        (df_clean["Smiles"].astype(str).str.strip() != "")
    ]
    df_clean["Standard Value"] = pd.to_numeric(
        df_clean["Standard Value"], errors="coerce"
    )
    df_clean = df_clean.dropna(
        subset=["Standard Value", "Smiles", "Document ChEMBL ID"]
    )

    total_unique_molecules = int(df_clean.shape[0])

    print(f"Rows after cleaning           : {len(df_clean):,}")
    print(f"Unique molecules (before threshold): {total_unique_molecules:,}")
    print(f"Standard types found          : {sorted(df_clean['Standard Type'].dropna().unique())}")

    return df_clean, total_unique_molecules


# ── Step 2: Per-standard-type processing ──────────────────────────────────────

def process_standard_type(
    df_clean: pd.DataFrame,
    stype: str,
    target_protein: str,
    potency_rules: dict = POTENCY_RULES,
    potency_thresholds: dict = POTENCY_THRESHOLDS,
    save_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Process one Standard Type: scaffold assignment, best-molecule selection,
    threshold filtering, and optional CSV export.

    Parameters
    ----------
    df_clean        : Cleaned DataFrame from load_and_clean().
    stype           : Standard Type string (e.g. "IC50").
    target_protein  : Used for output file naming.
    potency_rules   : Ascending/descending sort rules per stype.
    potency_thresholds : Threshold ranges per stype.
    save_csv        : Write the post-threshold best-per-doc CSV to disk.

    Returns
    -------
    best_post : Post-threshold best-per-document DataFrame.
    best_pre  : Pre-threshold best-per-document DataFrame.
    n_docs_passing : Number of documents passing threshold.
    n_mols_in_surviving : Unique molecule IDs in surviving documents.
    """
    print(f"\nProcessing: {stype}")
    temp_df = df_clean[df_clean["Standard Type"] == stype].copy()

    # Scaffold per document
    temp_df["Scaffold"] = temp_df["Smiles"].apply(get_scaffold)
    scaffold_map, count_map = {}, {}
    for doc_id, group in temp_df.groupby("Document ChEMBL ID"):
        count_map[doc_id] = len(group)
        scaffolds = group["Scaffold"].dropna()
        scaffold_map[doc_id] = (
            Counter(scaffolds).most_common(1)[0][0] if len(scaffolds) > 0 else None
        )

    # Best molecule per document
    ascending = potency_rules[stype]
    ranked_df = temp_df.sort_values("Standard Value", ascending=ascending)
    best_per_doc = ranked_df.groupby("Document ChEMBL ID", as_index=False).first()
    best_per_doc["Most Common Scaffold"] = best_per_doc["Document ChEMBL ID"].map(scaffold_map)
    best_per_doc["Number of Entries"] = best_per_doc["Document ChEMBL ID"].map(count_map)

    best_pre = best_per_doc.copy()

    # Apply threshold
    before = len(best_per_doc)
    best_post = best_per_doc[
        best_per_doc.apply(lambda row: passes_threshold(row, stype, potency_thresholds), axis=1)
    ].copy()
    after = len(best_post)

    print(f"  Documents before threshold : {before}")
    print(f"  Documents after threshold  : {after}")

    # Unique molecules in surviving documents
    surviving_doc_ids = set(best_post["Document ChEMBL ID"].unique())
    if surviving_doc_ids:
        n_mols = int(
            df_clean[
                (df_clean["Standard Type"] == stype) &
                (df_clean["Document ChEMBL ID"].isin(surviving_doc_ids))
            ]["Molecule ChEMBL ID"].nunique()
        )
    else:
        n_mols = 0

    print(f"  Unique mols in surviving docs : {n_mols}")

    os.makedirs(f"{target_protein}/csvs/chembl_activity", exist_ok=True)
    if save_csv:
        out_path = f"{target_protein}/csvs/chembl_activity/{target_protein}_chembl_most_potent_{stype}.csv"
        best_post.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    return best_post, best_pre, after, n_mols


def process_all_standard_types(
    df_clean: pd.DataFrame,
    target_protein: str,
    potency_rules: dict = POTENCY_RULES,
    potency_thresholds: dict = POTENCY_THRESHOLDS,
    save_csv: bool = True,
) -> tuple[dict, dict, dict, dict]:
    """
    Iterate over all Standard Types present in df_clean and call
    process_standard_type() for each known type.

    Returns
    -------
    all_best_post       : {stype: post-threshold DataFrame}
    all_best_pre        : {stype: pre-threshold DataFrame}
    stype_doc_counts    : {stype: n_docs_passing}
    stype_mol_counts    : {stype: n_mols_in_surviving_docs}
    """
    all_best_post, all_best_pre = {}, {}
    stype_doc_counts, stype_mol_counts = {}, {}

    for stype in df_clean["Standard Type"].dropna().unique():
        if stype not in potency_rules:
            print(f"Skipping unknown Standard Type: {stype}")
            continue
        post, pre, n_docs, n_mols = process_standard_type(
            df_clean, stype, target_protein,
            potency_rules, potency_thresholds, save_csv,
        )
        all_best_post[stype] = post
        all_best_pre[stype] = pre
        stype_doc_counts[stype] = n_docs
        stype_mol_counts[stype] = n_mols

    return all_best_post, all_best_pre, stype_doc_counts, stype_mol_counts


# ── Step 3: Summary counts ────────────────────────────────────────────────────

def compute_summary_counts(
    df_clean: pd.DataFrame,
    all_best_post: dict,
    total_unique_molecules: int,
) -> dict:
    """
    Compute the four headline summary counts.

    Returns
    -------
    dict with keys:
        total_unique_molecules, molecules_after_threshold,
        studies_passing, all_surviving_doc_ids
    """
    all_surviving_doc_ids = set()
    for stype_df in all_best_post.values():
        all_surviving_doc_ids.update(stype_df["Document ChEMBL ID"].unique())

    studies_passing = len(all_surviving_doc_ids)

    molecules_after = int(
        df_clean[
            df_clean["Document ChEMBL ID"].isin(all_surviving_doc_ids)
        ]["Molecule ChEMBL ID"].nunique()
    ) if all_surviving_doc_ids else 0

    print(f"\n{'='*60}")
    print(f"  FINAL COUNTS")
    print(f"{'='*60}")
    print(f"  ① Unique molecules before threshold : {total_unique_molecules:,}")
    print(f"  ④ Unique molecules after threshold  : {molecules_after:,}")
    print(f"  ② Unique studies passing threshold  : {studies_passing:,}")
    print(f"{'='*60}")

    return {
        "total_unique_molecules": total_unique_molecules,
        "molecules_after_threshold": molecules_after,
        "studies_passing": studies_passing,
        "all_surviving_doc_ids": all_surviving_doc_ids,
    }


# ── Step 4: Save summary report ───────────────────────────────────────────────

def save_summary_report(
    target_protein: str,
    summary_counts: dict,
    stype_doc_counts: dict,
    stype_mol_counts: dict,
    potency_thresholds: dict = POTENCY_THRESHOLDS,
) -> str:
    """
    Write the plain-text summary report to disk and return the report text.

    Parameters
    ----------
    target_protein   : Used for file path construction.
    summary_counts   : Output of compute_summary_counts().
    stype_doc_counts : {stype: n_docs} from process_all_standard_types().
    stype_mol_counts : {stype: n_mols} from process_all_standard_types().
    potency_thresholds : Threshold config for display.

    Returns
    -------
    report_text : The full report as a string.
    """
    total  = summary_counts["total_unique_molecules"]
    after  = summary_counts["molecules_after_threshold"]
    studies = summary_counts["studies_passing"]

    lines = [
        "=" * 60,
        f"  ChEMBL DATA SUMMARY — {target_protein}",
        "=" * 60,
        "",
        f"Molecules before threshold : {total:,}",
        f"Molecules after threshold  : {after:,}",
        f"Studies passing threshold  : {studies:,}",
        "",
        "-" * 60,
        f"  {'Standard Type':<14}  {'Docs (passed)':>14}  {'Mols in docs':>13}  Threshold",
        "-" * 60,
    ]

    for stype in sorted(stype_mol_counts):
        n_docs = stype_doc_counts.get(stype, 0)
        n_mols = stype_mol_counts.get(stype, 0)
        thresh = potency_thresholds.get(stype, {})
        if thresh.get("max") is not None:
            thresh_str = f"<= {thresh['max']:,} nM"
        elif thresh.get("min") is not None:
            thresh_str = f">= {thresh['min']:,}"
        else:
            thresh_str = "none"
        lines.append(f"  {stype:<14}  {n_docs:>14,}  {n_mols:>13,}  {thresh_str}")

    lines += [
        "-" * 60,
        "  Note: 'Mols in docs' = unique molecules in all rows of surviving",
        "  documents for that stype, not just the best-molecule per doc.",
        "  Same molecule in Activity + IC50 docs counts once in ② and ④.",
        "",
        "=" * 60,
        "End of report",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    report_path = f"{target_protein}/txts/{target_protein}_chembl_activity_summary_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSummary report saved: {report_path}")
    print(report_text)
    return report_text


# ── Step 5: Publications-per-year chart ───────────────────────────────────────

def plot_publications_per_year(
    all_best_pre: dict,
    target_protein: str,
    save_path: str | None = None,
) -> plt.Figure | None:
    """
    Plot unique ChEMBL source documents per year with a quadratic trend line.

    Parameters
    ----------
    all_best_pre   : Pre-threshold {stype: DataFrame} from process_all_standard_types().
    target_protein : Used for default save path if save_path is None.
    save_path      : Override output path (default: <protein>_chembl_publications_per_year.png).

    Returns
    -------
    matplotlib Figure, or None if no year data is available.
    """
    combined = pd.concat(all_best_pre.values(), ignore_index=True)

    if "Document Year" not in combined.columns:
        print("No 'Document Year' column — skipping chart.")
        return None

    year_series = (
        combined
        .drop_duplicates(subset=["Document ChEMBL ID"])["Document Year"]
        .dropna()
        .astype(int)
    )
    year_counts = year_series.value_counts().sort_index()
    years  = year_counts.index.to_numpy()
    counts = year_counts.values.astype(float)

    fig, ax = plt.subplots(figsize=(12.5, 4.8), facecolor="#F3F3F3")
    ax.set_facecolor("#EFEFEF")

    line_color  = "#C45A1A"
    trend_color = "#4B5563"
    grid_color  = "#D8D8D8"

    ax.plot(years, counts, color=line_color, linewidth=2.2,
            marker="o", markersize=5.5, markerfacecolor=line_color,
            markeredgewidth=0, zorder=4)
    ax.fill_between(years, counts, alpha=0.35, color="#FFFFFF", zorder=1)

    if len(years) >= 4:
        z = np.polyfit(years, counts, 2)
        x_trend = np.linspace(years.min(), years.max(), 300)
        ax.plot(x_trend, np.poly1d(z)(x_trend), linestyle="--",
                linewidth=1.8, color=trend_color, alpha=0.95,
                label="Trend", zorder=3)

    for x, y in zip(years, counts):
        ax.text(x, y + 0.12, str(int(y)), ha="center", va="bottom",
                fontsize=8.5, color="#2F2F2F")

    ax.set_title("ChEMBL Activity Publications per Year", fontsize=16,
                 fontweight="bold", color="#111111", loc="left", pad=10)
    ax.set_ylabel("Unique Documents", fontsize=11, color="#222222")
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax.set_ylim(0, max(counts) * 1.35)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9, color="#444444")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=9, colors="#444444")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color=grid_color, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.legend(frameon=True, facecolor="#F7F7F7", edgecolor="#CCCCCC",
              fontsize=9, loc="upper right")
    fig.text(0.5, 0.02, "Unique ChEMBL source documents per year.",
             ha="center", fontsize=10, color="#666666")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out = save_path or f"{target_protein}/images/{target_protein}_chembl_publications_per_year.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nChart saved: {out}")
    return fig


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_ch_activity_pipeline(
    target_protein: str,
    file_path: str | None = None,
    potency_rules: dict = POTENCY_RULES,
    potency_thresholds: dict = POTENCY_THRESHOLDS,
    save_csv: bool = True,
    show_report: bool = True,
) -> dict:
    """
    Run the complete ChEMBL activity processing pipeline end-to-end.

    Parameters
    ----------
    target_protein    : Protein identifier used in file paths.
    file_path         : Override input CSV path
                        (default: <protein>/<protein>_chembl.csv).
    potency_rules     : Override default sort-direction rules.
    potency_thresholds: Override default threshold ranges.
    save_csv          : Write per-stype best-per-doc CSVs to disk.
    show_report       : Print the summary report to stdout.

    Returns
    -------
    dict with keys:
        df_clean, total_unique_molecules,
        all_best_post, all_best_pre,
        stype_doc_counts, stype_mol_counts,
        summary_counts, report_text, fig_publications
    """
    csv_path = file_path or f"{target_protein}/{target_protein}_chembl.csv"

    # 1. Load & clean
    df_clean, total_unique = load_and_clean(csv_path)

    # 2. Process all standard types
    all_best_post, all_best_pre, stype_doc_counts, stype_mol_counts = (
        process_all_standard_types(
            df_clean, target_protein,
            potency_rules, potency_thresholds, save_csv,
        )
    )

    # 3. Summary counts
    summary_counts = compute_summary_counts(df_clean, all_best_post, total_unique)

    # 4. Report
    report_text = save_summary_report(
        target_protein, summary_counts,
        stype_doc_counts, stype_mol_counts, potency_thresholds,
    )

    # 5. Publications chart
    fig_pub = plot_publications_per_year(all_best_pre, target_protein)

    return {
        "df_clean": df_clean,
        "total_unique_molecules": total_unique,
        "all_best_post": all_best_post,
        "all_best_pre": all_best_pre,
        "stype_doc_counts": stype_doc_counts,
        "stype_mol_counts": stype_mol_counts,
        "summary_counts": summary_counts,
        "report_text": report_text,
        "fig_publications": fig_pub,
    }