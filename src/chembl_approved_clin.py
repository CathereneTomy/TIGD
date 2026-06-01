#======================================================================================
# APPROVED & CLINICAL MOLS
#======================================================================================
"""
chembl_selectivity_pipeline.py
================================
End-to-end pipeline for identifying potent, selective molecules against a
target protein from a ChEMBL CSV export.

Quick start
-----------
    from chembl_selectivity_pipeline import run_selectivity_pipeline

    df_all, df_selective = run_selectivity_pipeline(
        target_protein="EGFR",
        selectivity_delta_threshold=1.0,   # 10x selectivity (log scale)
    )
"""

from __future__ import annotations

import math
import time
from io import StringIO
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests
from rdkit import Chem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

CONCENTRATION_TYPES = {"IC50", "EC50", "Ki", "Kd", "XC50", "AC50", "ED50"}
LD50_TYPE           = {"LD50"}
PERCENTAGE_TYPES    = {"Inhibition", "Activity"}
ALL_TYPES           = CONCENTRATION_TYPES | LD50_TYPE | PERCENTAGE_TYPES

POTENCY_THRESHOLDS: dict[str, dict] = {
    "IC50":       {"max": 100,     "min": None},
    "EC50":       {"max": 100,     "min": None},
    "Ki":         {"max": 10,      "min": None},
    "Kd":         {"max": 10,      "min": None},
    "XC50":       {"max": 100,     "min": None},
    "AC50":       {"max": 100,     "min": None},
    "ED50":       {"max": 100,     "min": None},
    "Inhibition": {"max": None,    "min": 80},
    "Activity":   {"max": None,    "min": 80},
    "LD50":       {"max": None,    "min": 100_000},
}

UNIT_TO_NM: dict[str, float] = {
    "nM": 1,
    "uM": 1_000,
    "µM": 1_000,
    "mM": 1_000_000,
    "M":  1_000_000_000,
}

SELECTED_COLUMNS = [
    "Molecule ChEMBL ID", "Molecule Name", "Molecule Max Phase",
    "Molecular Weight", "Smiles", "Standard Type", "Standard Value",
    "Standard Units", "Data Validity Comment", "Assay ChEMBL ID",
    "Assay Description", "Assay Type", "BAO Label", "Assay Organism",
    "Document ChEMBL ID", "Document Year", "Target ChEMBL ID", "Target Name",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActivityRecord:
    target_chembl_id:  Optional[str]
    target_name:       Optional[str]
    target_organism:   Optional[str]
    standard_type:     Optional[str]
    standard_value:    Optional[float]
    standard_units:    Optional[str]
    assay_chembl_id:   Optional[str]
    assay_type:        Optional[str]
    pchembl_value:     Optional[float]
    is_ld50:           bool
    is_percentage:     bool
    document_chembl_id: Optional[str]
    document_link:     Optional[str]
    src_doc_id:        Optional[str]
    document_journal:  Optional[str]
    document_year:     Optional[str]


@dataclass
class TargetSummary:
    target_chembl_id:           str
    target_name:                Optional[str]
    target_organism:            Optional[str]
    best_pchembl:               Optional[float]
    best_pchembl_document_id:   Optional[str]   # doc that reported the best pChEMBL
    best_pchembl_document_link: Optional[str]   # direct link to that document
    standard_types:             list[str]
    num_assays:                 int
    document_ids:               list[str]
    document_links:             list[str]


# ---------------------------------------------------------------------------
# Step 1 — Load & pre-filter CSV
# ---------------------------------------------------------------------------

def _load_and_clean(target_protein: str) -> pd.DataFrame:
    """Read ChEMBL CSV, clean columns, and drop rows missing key fields."""
    file_path = f"{target_protein}/{target_protein}_chembl.csv"

    with open(file_path, "r", encoding="utf-8") as fh:
        content = fh.read().replace(",", " ")

    df = pd.read_csv(StringIO(content), sep=";")
    df = df[SELECTED_COLUMNS].copy()
    df = df.dropna(subset=["Standard Value", "Smiles"])
    df = df[
        df["Standard Value"].astype(str).str.strip().ne("") &
        df["Smiles"].astype(str).str.strip().ne("")
    ]
    df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
    df = df.dropna(subset=["Standard Value", "Smiles", "Document ChEMBL ID", "Molecule Name"])
    return df


def _passes_potency_threshold(standard_type: str, standard_value: float) -> bool:
    """Return True if the raw value falls within the acceptable potency range."""
    if standard_type not in ALL_TYPES or standard_value is None:
        return False
    t = POTENCY_THRESHOLDS.get(standard_type, {})
    if t.get("max") is not None and standard_value > t["max"]:
        return False
    if t.get("min") is not None and standard_value < t["min"]:
        return False
    return True


def _is_potent_row(row: pd.Series) -> bool:
    stype = row["Standard Type"]
    # Exclude types whose boolean rule is False (e.g. LD50, Inhibition, Activity)
    if stype in {"LD50", "Inhibition", "Activity"}:
        return False
    return _passes_potency_threshold(stype, row["Standard Value"])


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def _shortlist_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only potent rows, canonicalise SMILES, deduplicate.

    Fallback: if no molecule survives the potency filter, take the top 3
    molecules with the lowest Standard Value from rows whose Standard Units
    are nM (exact match, case-insensitive). This ensures the pipeline always
    has candidates to evaluate even for targets with few tight binders.
    """
    df_potent = df[df.apply(_is_potent_row, axis=1)].copy()

    if df_potent.empty:
        print(
            "⚠  No molecules passed the potency filter. "
            "Falling back to top-3 lowest-value nM compounds."
        )
        df_nm = df[
            df["Standard Units"].astype(str).str.strip().str.lower() == "nm"
        ].copy()

        if df_nm.empty:
            print("⚠  No nM rows found either — returning empty shortlist.")
            return pd.DataFrame(columns=df.columns)

        # Canonicalise and deduplicate before picking top-3 so we don't waste
        # slots on duplicate structures
        df_nm["canonical_smiles"] = df_nm["Smiles"].astype(str).apply(_canonicalize_smiles)
        df_nm = df_nm.dropna(subset=["canonical_smiles", "Molecule Max Phase"])
        df_nm = df_nm.drop_duplicates(subset=["canonical_smiles"])

        df_potent = (
            df_nm
            .sort_values("Standard Value", ascending=True)
            .head(3)
            .reset_index(drop=True)
        )
        print(
            f"   Selected {len(df_potent)} fallback molecule(s): "
            f"{df_potent['Molecule ChEMBL ID'].tolist()}"
        )
        return df_potent

    df_potent["canonical_smiles"] = df_potent["Smiles"].astype(str).apply(_canonicalize_smiles)
    df_potent = df_potent.dropna(subset=["canonical_smiles", "Molecule Max Phase"])
    df_potent = df_potent.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    return df_potent


# ---------------------------------------------------------------------------
# Step 2 — ChEMBL API helpers
# ---------------------------------------------------------------------------

def _compute_pchembl(
    standard_type: Optional[str],
    standard_value: Optional[float],
    standard_units: Optional[str],
) -> Optional[float]:
    """Convert raw activity to pChEMBL = -log10(value_in_molar)."""
    if standard_type not in (CONCENTRATION_TYPES | LD50_TYPE):
        return None
    if not standard_value or standard_value <= 0:
        return None
    multiplier = UNIT_TO_NM.get(standard_units)
    if multiplier is None:
        return None
    value_nM = standard_value * multiplier
    return round(-math.log10(value_nM * 1e-9), 3)


def _fetch_all_activities(chembl_id: str, target_of_interest: str = None) -> list[dict]:
    """
    Fetch top 50 most potent activity records for a molecule from ChEMBL,
    then exclude the target of interest client-side and return top 10.

    Fetching 50 gives enough headroom to find off-targets even when the
    on-target dominates the top slots (e.g. Deucravacitinib vs TYK2).

    target_chembl_id__not_in is NOT used — ChEMBL returns 500 for that filter.
    """
    url = (
        f"{CHEMBL_BASE}/activity.json"
        f"?molecule_chembl_id={chembl_id}"
        f"&standard_type__in=IC50,EC50,Ki,Kd,XC50,AC50,ED50"
        f"&standard_units=nM"
        f"&standard_value__isnull=false"
        f"&standard_value__gt=0"
        f"&order_by=standard_value"
        f"&limit=50&offset=0"
    )

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} for {url}")
            return []
        activities = r.json().get("activities", [])

        # Exclude on-target client-side
        if target_of_interest:
            activities = [
                a for a in activities
                if str(a.get("target_chembl_id", "")).strip().upper()
                != str(target_of_interest).strip().upper()
            ]

        return activities[:10]   # top 10 off-targets by potency

    except requests.exceptions.Timeout:
        print(f"    Timeout for {chembl_id}")
        return []
    except Exception as exc:
        print(f"    Request error: {exc}")
        return []


def _parse_activity(act: dict) -> ActivityRecord:
    """Extract and normalise fields from a raw ChEMBL activity record."""
    try:
        std_value = float(act.get("standard_value") or 0) or None
    except (TypeError, ValueError):
        std_value = None

    std_type  = act.get("standard_type")
    std_units = act.get("standard_units")

    raw_pchembl = act.get("pchembl_value")
    try:
        pchembl = float(raw_pchembl) if raw_pchembl is not None else None
    except (TypeError, ValueError):
        pchembl = None
    pchembl = pchembl or _compute_pchembl(std_type, std_value, std_units)

    doc_chembl_id = act.get("document_chembl_id")
    return ActivityRecord(
        target_chembl_id   = act.get("target_chembl_id"),
        target_name        = act.get("target_pref_name"),
        target_organism    = act.get("target_organism"),
        standard_type      = std_type,
        standard_value     = std_value,
        standard_units     = std_units,
        assay_chembl_id    = act.get("assay_chembl_id"),
        assay_type         = act.get("assay_type"),
        pchembl_value      = pchembl,
        is_ld50            = std_type in LD50_TYPE,
        is_percentage      = std_type in PERCENTAGE_TYPES,
        document_chembl_id = doc_chembl_id,
        document_link      = (
            f"https://www.ebi.ac.uk/chembl/document_report_card/{doc_chembl_id}"
            if doc_chembl_id else None
        ),
        src_doc_id         = act.get("src_id"),
        document_journal   = act.get("document_journal"),
        document_year      = act.get("document_year"),
    )


# ---------------------------------------------------------------------------
# Step 3 — On / off target split and summarisation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Target-type cache  (single-protein guard for off-target filtering)
# ---------------------------------------------------------------------------

# Module-level cache: target_chembl_id (upper) -> bool (True = SINGLE PROTEIN)
_target_type_cache: dict[str, bool] = {}

def _warm_target_type_cache(target_ids: list[str]) -> None:
    """
    Batch-fetch target types for all IDs not yet in the cache.
    Uses ChEMBL's target__in filter: one paginated request resolves up to
    1000 targets at a time, so a molecule with 50 off-targets costs 1 call
    instead of 50.  Results are stored in _target_type_cache.
    """
    missing = [
        tid for tid in target_ids
        if str(tid).strip().upper() not in _target_type_cache
    ]
    if not missing:
        return

    # Process in chunks of 100 (safe URL length)
    chunk_size = 100
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i : i + chunk_size]
        ids_param = ";".join(chunk)
        url = (
            f"{CHEMBL_BASE}/target.json"
            f"?target_chembl_id__in={ids_param}&limit={chunk_size}&only=target_chembl_id,target_type"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            for t in resp.json().get("targets", []):
                key   = str(t.get("target_chembl_id", "")).strip().upper()
                ttype = str(t.get("target_type", "")).strip().upper()
                _target_type_cache[key] = (ttype == "SINGLE PROTEIN")
        except Exception:
            pass  # leave missing IDs absent; _is_single_protein_target will default True

    # Any IDs the API didn't return (unknown targets) → default True
    for tid in missing:
        key = str(tid).strip().upper()
        if key not in _target_type_cache:
            _target_type_cache[key] = True


def _is_single_protein_target(target_chembl_id: str) -> bool:
    """
    Return True iff the ChEMBL target is classified as SINGLE PROTEIN.
    Call _warm_target_type_cache() with a batch of IDs first for speed.
    Falls back to a single API call if the ID is not yet cached.
    Defaults to True on any failure so on-target data is never silently lost.
    """
    key = str(target_chembl_id).strip().upper()
    if key in _target_type_cache:
        return _target_type_cache[key]

    # Single fallback lookup (should rarely be needed after batch warm)
    try:
        url  = f"{CHEMBL_BASE}/target/{key}.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        ttype = str(resp.json().get("target_type", "")).strip().upper()
        result = ttype == "SINGLE PROTEIN"
    except Exception:
        result = True

    _target_type_cache[key] = result
    return result


def _fetch_on_target_activities(chembl_id: str, target_chembl_id: str) -> list[ActivityRecord]:
    """
    Fetch activity records for a molecule directly against a specific target
    using ChEMBL's combined molecule + target filter.

    This avoids pulling the full activity set for the molecule and filtering
    afterwards — the API returns only on-target records in one paginated call:
        GET /activity.json?molecule_chembl_id=X&target_chembl_id=Y
    """
    activities = []
    url = (
        f"{CHEMBL_BASE}/activity.json"
        f"?molecule_chembl_id={chembl_id}"
        f"&target_chembl_id={target_chembl_id}"
        f"&limit=100&offset=0"
    )

    while url:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"    HTTP {r.status_code} fetching on-target activities: {url}")
                break
            data = r.json()
        except Exception as exc:
            print(f"    Request error (on-target fetch): {exc}")
            break

        activities.extend(data.get("activities", []))
        next_page = data.get("page_meta", {}).get("next")
        url = f"https://www.ebi.ac.uk{next_page}" if next_page else None
        time.sleep(0.2)

    parsed = [_parse_activity(a) for a in activities]

    if not parsed:
        print(f"    ⚠ No on-target records found for {chembl_id} vs {target_chembl_id}")

    return parsed


def _split_on_off_targets(
    chembl_id: str,
    target_of_interest: str,
) -> tuple[list[ActivityRecord], list[ActivityRecord]]:
    """
    Fetch on-target and off-target activity records separately.

    On-target  — fetched directly via _fetch_on_target_activities using the
                 combined molecule + target filter; no post-filtering needed.
    Off-target — fetched via _fetch_all_activities (all activities for the
                 molecule), then filtered to single-protein targets only,
                 excluding the primary target and weak/unresolvable hits.
    """
    target_norm = str(target_of_interest).strip().upper()

    # ── On-target: dedicated fetch, no filtering required ────────────────────
    on_hits = _fetch_on_target_activities(chembl_id, target_norm)

    # ── Off-target: full fetch → single-protein filter → potency filter ──────
    raw        = _fetch_all_activities(chembl_id)
    parsed_all = [_parse_activity(a) for a in raw]

    unique_target_ids = [
        str(p.target_chembl_id or "").strip().upper()
        for p in parsed_all
        if p.target_chembl_id
    ]
    _warm_target_type_cache(unique_target_ids)
    single_protein_ids = {
        tid for tid in unique_target_ids
        if _is_single_protein_target(tid)
    }

    off_hits = [
        p for p in parsed_all
        if str(p.target_chembl_id or "").strip().upper() != target_norm
        and str(p.target_chembl_id or "").strip().upper() in single_protein_ids
        and _passes_potency_threshold(p.standard_type, p.standard_value)
        and _get_gene_label(str(p.target_chembl_id or "").strip().upper()) is not None
    ]

    return on_hits, off_hits


def _summarise_target(hits: list[ActivityRecord], target_chembl_id: str) -> TargetSummary:
    """Collapse multiple assay records for one target into a single summary."""
    potency_hits = [
        h for h in hits
        if not h.is_ld50 and not h.is_percentage and h.pchembl_value is not None
    ]
    best_pchembl = max((h.pchembl_value for h in potency_hits), default=None)

    # Find the specific hit that reported the best pChEMBL value
    best_hit = next(
        (h for h in potency_hits if h.pchembl_value == best_pchembl),
        None,
    )

    doc_ids   = sorted({h.document_chembl_id for h in hits if h.document_chembl_id})
    doc_links = sorted({h.document_link       for h in hits if h.document_link})

    return TargetSummary(
        target_chembl_id           = target_chembl_id,
        target_name                = hits[0].target_name,
        target_organism            = hits[0].target_organism,
        best_pchembl               = best_pchembl,
        best_pchembl_document_id   = best_hit.document_chembl_id if best_hit else None,
        best_pchembl_document_link = best_hit.document_link      if best_hit else None,
        standard_types             = sorted({h.standard_type for h in hits if h.standard_type}),
        num_assays                 = len(hits),
        document_ids               = doc_ids,
        document_links             = doc_links,
    )


def _summarise_off_targets(off_hits: list[ActivityRecord]) -> list[TargetSummary]:
    """Group off-target hits by target ID and summarise each group."""
    grouped: dict[str, list[ActivityRecord]] = {}
    for h in off_hits:
        grouped.setdefault(h.target_chembl_id, []).append(h)

    summaries = [
        _summarise_target(hits, tid)
        for tid, hits in grouped.items()
    ]
    summaries.sort(
        key=lambda s: s.best_pchembl if s.best_pchembl is not None else -float("inf"),
        reverse=True,
    )
    return summaries


# ---------------------------------------------------------------------------
# Step 4 — Selectivity calculation
# ---------------------------------------------------------------------------

def _compute_selectivity(
    on_summary: Optional[TargetSummary],
    off_summaries: list[TargetSummary],
    threshold: float,
) -> tuple[Optional[float], bool, str]:
    """
    Selectivity delta = on_pchembl - best_off_pchembl.
    Returns (delta, is_selective, reason).
    """
    if on_summary is None or on_summary.best_pchembl is None:
        return None, False, "no_on_target_pchembl"

    valid_off = [s.best_pchembl for s in off_summaries if s.best_pchembl is not None]
    if not valid_off:
        return None, True, "no_off_target_pchembl_data"

    best_off  = max(valid_off)
    delta     = round(on_summary.best_pchembl - best_off, 3)
    return delta, delta >= threshold, f"delta={delta}"


# ---------------------------------------------------------------------------
# Step 5 — Build result row
# ---------------------------------------------------------------------------

def _build_record(
    row: pd.Series,
    on_summary: Optional[TargetSummary],
    off_summaries: list[TargetSummary],
    sel_delta: Optional[float],
    is_selective: bool,
    sel_reason: str,
) -> dict:
    on_doc_ids   = on_summary.document_ids   if on_summary else []
    on_doc_links = on_summary.document_links if on_summary else []

    off_doc_by_target  = {s.target_name: s.document_ids  for s in off_summaries if s.document_ids}
    off_doc_links_flat = sorted({link for s in off_summaries for link in s.document_links})

    # Top off-target: the one with highest pchembl (list is already sorted desc)
    top_off = off_summaries[0] if off_summaries else None

    # canonical_smiles: prefer existing canonical_smiles col, fall back to Smiles
    smiles = row.get("canonical_smiles") or row.get("Smiles") or row.get("smiles")

    return {
        # Identity
        "Molecule ChEMBL ID":    row["Molecule ChEMBL ID"],
        "Molecule Name":         row["Molecule Name"],
        "Molecule Max Phase":    row["Molecule Max Phase"],
        "canonical_smiles":      smiles,
        # On-target
        "on_target_chembl_id":   on_summary.target_chembl_id if on_summary else None,
        "on_target_name":        on_summary.target_name      if on_summary else None,
        "on_target_best_pchembl":on_summary.best_pchembl     if on_summary else None,
        "on_target_std_types":   on_summary.standard_types   if on_summary else None,
        "on_target_num_assays":  on_summary.num_assays       if on_summary else None,
        "on_target_document_ids":              on_doc_ids,
        "on_target_document_links":            on_doc_links,
        "on_target_best_pchembl_document_id":  on_summary.best_pchembl_document_id   if on_summary else None,
        "on_target_best_pchembl_document_link":on_summary.best_pchembl_document_link if on_summary else None,
        # Off-target
        "off_target_count":      len(off_summaries),
        "off_targets":           off_summaries,
        "off_target_names":      [s.target_name      for s in off_summaries],
        "off_target_chembl_ids": [s.target_chembl_id for s in off_summaries],
        "off_target_best_pchembl":{s.target_name: s.best_pchembl for s in off_summaries},
        # Top off-target (the most potent one)
        "top_off_target_name":                    top_off.target_name                  if top_off else None,
        "top_off_target_chembl_id":               top_off.target_chembl_id             if top_off else None,
        "top_off_target_pchembl":                 top_off.best_pchembl                 if top_off else None,
        "top_off_target_best_pchembl_document_id":  top_off.best_pchembl_document_id   if top_off else None,
        "top_off_target_best_pchembl_document_link":top_off.best_pchembl_document_link if top_off else None,
        "off_target_document_ids_by_target": off_doc_by_target,
        "off_target_document_links":         off_doc_links_flat,
        # Selectivity
        "selectivity_delta":      sel_delta,
        "is_selective":           is_selective,
        "selectivity_reason":     sel_reason,
        "has_on_target_activity": on_summary is not None,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_selectivity_pipeline(
    target_protein: str,
    selectivity_delta_threshold: float = None,
    sleep_between_molecules: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load CSV → shortlist potent molecules → query ChEMBL for
    on/off target activity → compute selectivity → return results.

    Parameters
    ----------
    target_protein : str
        Folder name (and file prefix) for the ChEMBL CSV, e.g. "EGFR".
        Expected file: ``<target_protein>/<target_protein>_chembl.csv``
    selectivity_delta_threshold : float
        Minimum pChEMBL delta (on – off) required for a molecule to be
        considered selective. Default 1.0 ≈ 10x; use 2.0 for 100x.
    sleep_between_molecules : float
        Seconds to wait between API calls (default 0.3 s).

    Returns
    -------
    df_all : pd.DataFrame
        All evaluated molecules with full selectivity annotations.
    df_selective : pd.DataFrame
        Subset of df_all containing only selective molecules.
    """
    # --- Load & shortlist ---
    df_raw        = _load_and_clean(target_protein)
    df_shortlisted = _shortlist_molecules(df_raw)
    total         = len(df_shortlisted)
    print(f"Shortlisted {total} unique, potent molecules for {target_protein}")

    # --- Per-molecule API loop ---
    records = []
    for i, row in df_shortlisted.iterrows():
        chembl_id = row["Molecule ChEMBL ID"]
        mol_name  = row["Molecule Name"]
        target_id = row["Target ChEMBL ID"]
        print(f"[{i + 1}/{total}] {chembl_id} — {mol_name}")

        try:
            on_hits, off_hits = _split_on_off_targets(chembl_id, target_id)
        except Exception as exc:
            print(f"  ⚠ Failed: {exc}")
            on_hits, off_hits = [], []

        on_summary   = _summarise_target(on_hits, target_id) if on_hits else None
        off_summaries = _summarise_off_targets(off_hits)
        print(f"Selectivity threshold: {selectivity_delta_threshold}")
        sel_delta, is_selective, sel_reason = _compute_selectivity(
            on_summary, off_summaries, selectivity_delta_threshold
        )

        records.append(_build_record(
            row, on_summary, off_summaries, sel_delta, is_selective, sel_reason
        ))
        time.sleep(sleep_between_molecules)

    # --- Build & sort DataFrame ---
    if not records:
        print("⚠ No records collected — check that the CSV loaded correctly and "
              "that at least one molecule passed the potency filter.")
        empty = pd.DataFrame()
        return empty, empty

    df_all = pd.DataFrame(records)
    df_all = (
        df_all
        .assign(
            _sort_no_activity=df_all["has_on_target_activity"].map({True: 0, False: 1}),
            _sort_selective=df_all["is_selective"].map({True: 0, False: 1}),
            _sort_potency=-df_all["on_target_best_pchembl"].fillna(-float("inf")),
        )
        .sort_values(["_sort_no_activity", "_sort_selective", "_sort_potency"])
        .drop(columns=["_sort_no_activity", "_sort_selective", "_sort_potency"])
        .reset_index(drop=True)
    )

    # --- Selective subset ---
    def _is_selective_molecule(r: pd.Series) -> bool:
        if not r["has_on_target_activity"]:
            return False
        if r["off_target_count"] == 0:
            return True          # Tier 1: clean profile
        return r["is_selective"] # Tier 2: pChEMBL delta meets threshold

    df_selective = df_all[df_all.apply(_is_selective_molecule, axis=1)].copy()
    print(f"\n✅ Selective molecules: {len(df_selective)} / {len(df_all)}")

    # --- Pretty-print selective results ---
    display_cols = [
        "Molecule Name", "Molecule Max Phase",
        "on_target_name", "on_target_best_pchembl",
        "off_target_count", "off_target_names",
        "selectivity_delta", "is_selective",
        "on_target_document_ids", "on_target_document_links",
        "off_target_document_ids_by_target", "off_target_document_links",
    ]
    print(df_selective[display_cols].to_string())

    return df_all, df_selective


# ---------------------------------------------------------------------------
# Clinical merge
# ---------------------------------------------------------------------------

def merge_clinical_into_pipeline(
    clinical_df: pd.DataFrame,
    df_all: pd.DataFrame,
    target_chembl_id: str,
    selectivity_delta_threshold: float = None,
    sleep_between_molecules: float = 0.3,
) -> pd.DataFrame:
    """
    Ensure every molecule in ``clinical_df`` is represented in ``df_all``.

    For molecules whose ChEMBL ID is already in ``df_all`` the existing row is
    kept as-is (no duplicate API calls).  For molecules that are missing, the
    full on/off-target selectivity calculation is run and the result is appended.

    Unified column schema
    ─────────────────────
        molecule_chembl_id  ← Molecule ChEMBL ID  / molecule_chembl_id
        pref_name       ← Molecule Name        / pref_name
        max_phase           ← Molecule Max Phase   / max_phase
        canonical_smiles    ← canonical_smiles     / canonical_smiles
        + all on/off/selectivity columns from df_all

    Columns that exist only in clinical_df (molecule_type, clinical_status,
    mechanism_of_action, action_type, mol_wt) are intentionally dropped —
    the merged df contains only columns that df_all defines.
    """

    # 1. Normalise df_all column names
    rename_from_pipeline = {
        "Molecule ChEMBL ID": "molecule_chembl_id",
        "Molecule Name":      "pref_name",
        "Molecule Max Phase": "max_phase",
        # Smiles -> canonical_smiles if not already renamed by _build_record
        "Smiles":             "canonical_smiles",
    }
    df_pipeline = df_all.rename(columns={k: v for k, v in rename_from_pipeline.items()
                                          if k in df_all.columns}).copy()

    # 2. Early-exit if clinical_df is empty
    if clinical_df is None or clinical_df.empty:
        print("clinical_df is empty — returning pipeline results unchanged.")
        return df_pipeline

    # 3. Normalise clinical_df column names
    df_clin = clinical_df.copy()

    # 4. Find which clinical molecules are missing from df_all
    pipeline_ids = set(df_pipeline["molecule_chembl_id"].astype(str).str.strip().str.upper())
    clin_ids     = df_clin["molecule_chembl_id"].astype(str).str.strip().str.upper()
    missing_mask = ~clin_ids.isin(pipeline_ids)
    df_missing   = df_clin[missing_mask].copy()

    print(
        f"clinical_df has {len(df_clin)} molecules; "
        f"{missing_mask.sum()} not in df_all -> will calculate selectivity for those."
    )

    # 5. Run selectivity for missing molecules
    new_records = []
    total = len(df_missing)
    for i, row in enumerate(df_missing.itertuples(index=False), 1):
        chembl_id = str(row.molecule_chembl_id).strip()
        mol_name  = getattr(row, "pref_name", None) or chembl_id
        print(f"  [{i}/{total}] {chembl_id} - {mol_name}")

        try:
            on_hits, off_hits = _split_on_off_targets(chembl_id, target_chembl_id)
        except Exception as exc:
            print(f"    Warning: Failed: {exc}")
            on_hits, off_hits = [], []

        on_summary    = _summarise_target(on_hits, target_chembl_id) if on_hits else None
        off_summaries = _summarise_off_targets(off_hits)
        sel_delta, is_selective, sel_reason = _compute_selectivity(
            on_summary, off_summaries, selectivity_delta_threshold
        )

        synthetic_row = pd.Series({
            "Molecule ChEMBL ID": chembl_id,
            "Molecule Name":      mol_name,
            "Molecule Max Phase": getattr(row, "max_phase", None),
        })
        rec = _build_record(synthetic_row, on_summary, off_summaries,
                            sel_delta, is_selective, sel_reason)

        # Rename to unified schema
        rec["molecule_chembl_id"] = rec.pop("Molecule ChEMBL ID")
        rec["pref_name"]          = rec.pop("Molecule Name")
        rec["max_phase"]          = rec.pop("Molecule Max Phase")
        rec["canonical_smiles"]   = getattr(row, "canonical_smiles", None)

        new_records.append(rec)
        time.sleep(sleep_between_molecules)

    # 6. Combine and sort
    df_new    = pd.DataFrame(new_records) if new_records else pd.DataFrame()
    df_merged = pd.concat([df_pipeline, df_new], ignore_index=True)
    df_merged = (
        df_merged
        .sort_values(
            ["max_phase", "on_target_best_pchembl"],
            ascending=[False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    print(
        f"\nMerged DataFrame: {len(df_merged)} molecules total "
        f"({len(df_pipeline)} from pipeline + {len(df_new)} from clinical)."
    )
    return df_merged



# ---------------------------------------------------------------------------
# Gene-name resolution for target / off-target labels
# ---------------------------------------------------------------------------

# Words that appear as first-word of ChEMBL pref_name but are NOT gene symbols.
# Any resolved label matching these is treated as unresolvable.
_INVALID_GENE_LABELS = {
    "BLOOD", "UNCHECKED", "UNCHARACTERIZED", "UNREVIEWED", "PUTATIVE",
    "ORGANISM", "CELL", "TISSUE", "WHOLE", "UNDEFINED", "UNKNOWN",
    "SELECTIVITY", "PROTEIN", "CYTOCHROME", "NUCLEAR", "MEMBRANE",
}

def _chembl_target_to_gene(target_chembl_id: str) -> Optional[str]:
    """
    Resolve a ChEMBL target ID to a proper HGNC gene symbol.
    Returns None if no valid gene symbol can be found — callers must
    handle None and skip/filter those entries rather than showing junk.
    """
    if not target_chembl_id:
        return None
    try:
        url  = f"{CHEMBL_BASE}/target/{target_chembl_id}.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Priority 1: explicit GENE_SYMBOL or GENE_NAME synonym
        for comp in data.get("target_components", []):
            for syn in comp.get("target_component_synonyms", []):
                if syn.get("syn_type") in ("GENE_SYMBOL", "GENE_NAME"):
                    gene = syn.get("component_synonym", "").strip()
                    if gene and gene.upper() not in _INVALID_GENE_LABELS:
                        return gene.upper()

        # Priority 2: approved_symbol from component directly
        for comp in data.get("target_components", []):
            sym = comp.get("component_synonym", "").strip()
            if sym and sym.upper() not in _INVALID_GENE_LABELS:
                return sym.upper()

    except Exception:
        pass

    # No valid gene symbol found — return None so callers can filter it out
    return None


# Cache so we don't re-query the same target repeatedly across molecules
_gene_cache: dict[str, str] = {}

def _get_gene_label(target_chembl_id: str) -> Optional[str]:
    """
    Cached wrapper around _chembl_target_to_gene.
    Returns None if no valid gene symbol can be resolved — callers
    should filter out None values rather than displaying them.
    """
    key = str(target_chembl_id).strip().upper()
    if key not in _gene_cache:
        _gene_cache[key] = _chembl_target_to_gene(key)
    return _gene_cache[key]


# ---------------------------------------------------------------------------
# Human-readable selectivity column builder
# ---------------------------------------------------------------------------

def _build_selectivity_label(row: pd.Series) -> str:
    """
    Produce a compact selectivity summary string for one molecule row.
    Uses ChEMBL IDs (stored in on_target_chembl_id / off_target_chembl_ids)
    to resolve gene symbols — never the raw long protein name strings.

    Examples
    --------
    Selective (10x) vs TYK2 (pC=8.30) | off: JAK1 (pC=7.20), JAK2 (pC=6.80)
    Not selective vs PCSK9 (pC=6.70) | top off-target: TIMP3 (pC=8.72)
    No on-target activity found
    """
    if not row.get("has_on_target_activity"):
        return "No on-target activity found"

    on_pchembl       = row.get("on_target_best_pchembl")
    on_chembl_id     = row.get("on_target_chembl_id") or ""
    on_gene          = row.get("on_target_gene") or _get_gene_label(on_chembl_id)
    sel_delta        = row.get("selectivity_delta")
    is_sel           = row.get("is_selective", False)
    sel_reason       = row.get("selectivity_reason", "")

    # off_target_best_pchembl is {protein_name: pchembl}; pair with gene labels
    off_pchembl_map  = row.get("off_target_best_pchembl") or {}
    off_chembl_ids   = row.get("off_target_chembl_ids") or []
    off_gene_list    = row.get("off_target_genes") or []

    # Build a name->gene mapping (index matches off_target_names order)
    off_names        = row.get("off_target_names") or []
    name_to_gene     = dict(zip(off_names, off_gene_list)) if off_gene_list else {}

    # On-target fragment
    on_str = f"{on_gene} (pC={on_pchembl:.2f})" if on_pchembl else on_gene

    # Off-target fragments sorted by pchembl descending
    off_parts = []
    for t_name, pc in sorted(off_pchembl_map.items(), key=lambda x: (x[1] or 0), reverse=True):
        gene   = name_to_gene.get(t_name) or t_name
        pc_str = f"pC={pc:.2f}" if pc is not None else "pC=?"
        off_parts.append(f"{gene} ({pc_str})")

    off_str = ", ".join(off_parts) if off_parts else "none identified"

    if sel_reason == "no_off_target_pchembl_data":
        return f"Selective (clean profile) vs {on_str} | no off-target activity detected"

    if is_sel:
        fold = round(10 ** sel_delta) if sel_delta is not None else "?"
        return f"Selective ({fold}x) vs {on_str} | off: {off_str}"
    else:
        top_name = row.get("top_off_target_name") or ""
        top_gene = name_to_gene.get(top_name) or top_name
        top_pc   = row.get("top_off_target_pchembl")
        top_str  = f"{top_gene} (pC={top_pc:.2f})" if top_pc else top_gene
        return f"Not selective vs {on_str} | top off-target: {top_str} | all off: {off_str}"


# ---------------------------------------------------------------------------
# Master enrichment function: merge clinical + run PDB enrichment
# ---------------------------------------------------------------------------

def build_enriched_clinical_df(
    clinical_df: pd.DataFrame,
    df_all: pd.DataFrame,
    target_chembl_id: str,
    human_uniprot: str,
    mouse_uniprot: str,
    chembl_target: dict,
    selectivity_delta_threshold: float = None,
    sleep_between_molecules: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple], dict]:
    """
    One-stop function that:
      1. Merges clinical_df with df_all (running selectivity for any missing molecules)
      2. Adds a human-readable ``selectivity_summary`` column using gene labels
      3. Runs PDB enrichment (human/mouse PDB IDs, off-target map) on the merged df
      4. Returns the same tuple signature as ``enrich_with_pdb`` so existing
         ``save_ids_and_molecules`` / ``show_selector`` calls need no changes.

    Parameters
    ----------
    clinical_df              : output of get_clinical_molecules() — may be empty
    df_all                   : output of run_selectivity_pipeline()
    target_chembl_id         : e.g. "CHEMBL2364"
    human_uniprot / mouse_uniprot : UniProt IDs for PDB lookup
    chembl_target            : full ChEMBL target dict (for top-5 PDB resolution)
    selectivity_delta_threshold : pChEMBL delta threshold (default 1.0 = 10x)
    sleep_between_molecules  : API throttle in seconds

    Returns
    -------
    enriched_df      — merged df with PDB columns + selectivity_summary
    pdb_detail_df    — long-form one row per (molecule, species, ligand, pdb)
    top5_target_pdbs — list of (pdb_id, resolution)
    offtarget_map    — {uniprot_id: {organism, ligands: {chembl_id: {pdb_id}}}}
    """

    # ── Step 1: merge clinical into pipeline df ──────────────────────────────
    print("\nStep 1/3 — Merging clinical molecules into selectivity results...")
    merged_df = merge_clinical_into_pipeline(
        clinical_df=clinical_df,
        df_all=df_all,
        target_chembl_id=target_chembl_id,
        selectivity_delta_threshold=selectivity_delta_threshold,
        sleep_between_molecules=sleep_between_molecules,
    )

    if merged_df.empty:
        print("merged_df is empty")
        return merged_df, pd.DataFrame(), [], {}

    # ── Step 2: add selectivity_summary column ───────────────────────────────
    print("\nStep 2/3 — Building selectivity labels (resolving gene names)...")

    # Resolve gene symbols by ChEMBL ID (not by long protein name strings).
    # Pre-warm the cache for the primary target — same ID on every row.
    _get_gene_label(target_chembl_id)

    # on_target_chembl_id was added by _build_record; resolve to gene symbol.
    merged_df["on_target_gene"] = merged_df["on_target_chembl_id"].apply(
        lambda x: _get_gene_label(str(x).strip()) if pd.notna(x) and x else None
    )

    # off_target_chembl_ids is a list column; resolve each entry.
    # Filter out None results — targets with no resolvable gene symbol are
    # excluded from the gene list entirely rather than shown as None/junk.
    merged_df["off_target_genes"] = merged_df["off_target_chembl_ids"].apply(
        lambda ids: [
            g for g in (_get_gene_label(str(i).strip()) for i in (ids or []))
            if g is not None
        ]
    )

    # top off-target gene — None if unresolvable (renderer shows "-")
    merged_df["top_off_target_gene"] = merged_df["top_off_target_chembl_id"].apply(
        lambda x: _get_gene_label(str(x).strip()) if pd.notna(x) and x else None
    )

    merged_df["selectivity_summary"] = merged_df.apply(_build_selectivity_label, axis=1)

    # ── Step 3: PDB enrichment ───────────────────────────────────────────────
    print("\nStep 3/3 — Running PDB enrichment...")
    enriched_df, pdb_detail_df, top5_target_pdbs, offtarget_map = enrich_with_pdb(
        merged_df, human_uniprot, mouse_uniprot, chembl_target
    )

    print("\nDone. Enriched df shape:", enriched_df.shape)
    return enriched_df, pdb_detail_df, top5_target_pdbs, offtarget_map

# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------


#==============================================================================================
# OLD APPROVED MOLS
#==============================================================================================

def _pubchem_to_ligands(name: str) -> list:
    """Resolve a compound name/ID via PubChem and extract 3-letter ligand IDs."""
    try:
        cid_resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON",
            timeout=10,
        )
        cid_resp.raise_for_status()

        cids = cid_resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return []

        cid = cids[0]

        syn_resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/RegistryID/JSON",
            timeout=10,
        )
        syn_resp.raise_for_status()

        synonyms = (
            syn_resp.json()
            .get("InformationList", {})
            .get("Information", [{}])[0]
            .get("RegistryID", [])
        )

        return sorted(
            {
                s.strip()
                for s in synonyms
                if len(s.strip()) == 3 and s.strip().isupper()
            }
        )

    except Exception:
        return []


def _get_alternate_chembl_ids(chembl_id: str) -> list:
    """Fetch alternate/related ChEMBL IDs."""
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        data = resp.json()

        alternates = set()

        # Parent molecule
        parent = data.get("molecule_hierarchy", {}).get("parent_chembl_id")
        if parent:
            alternates.add(parent)

        # Cross references
        for xref in data.get("cross_references", []):
            xref_id = xref.get("xref_id")
            if xref_id and str(xref_id).startswith("CHEMBL"):
                alternates.add(xref_id)

        alternates.discard(chembl_id)

        return list(alternates)

    except Exception:
        return []


def _chembl_to_ligand_ids(chembl_id: str) -> list:
    """ChEMBL ID -> PubChem CID -> 3-letter PDB ligand candidates."""

    # Try original ID first
    ligands = _pubchem_to_ligands(chembl_id)
    if ligands:
        return ligands

    # Fallback to alternate IDs
    for alt_id in _get_alternate_chembl_ids(chembl_id):
        ligands = _pubchem_to_ligands(alt_id)
        if ligands:
            return ligands

    return []



import requests
import pandas as pd
from collections import defaultdict

# ── PDB API helpers ────────────────────────────────────────────────────────────


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def _uniprot_pdbs(uniprot_id: str) -> set:
    query = {
        "query": {
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers"
                    ".reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    r = requests.post(SEARCH_URL, json=query, timeout=15)
    r.raise_for_status()
    print({x["identifier"] for x in r.json().get("result_set", [])})
    return {x["identifier"] for x in r.json().get("result_set", [])}



GRAPHQL_URL = "https://data.rcsb.org/graphql"
def graphql(query: str) -> dict:
    """Send a GraphQL query and return parsed JSON data."""
    resp = requests.post(GRAPHQL_URL, json={"query": query}, timeout=15)
    resp.raise_for_status()
    return resp.json()

def get_uniprots_for_pdb(pdb_id: str, target_uids: set) -> list[dict]:
    """
    Given a PDB ID, return all UniProt accessions of the proteins in it,
    excluding the target's own UniProt IDs.
    """
    pdb_id = pdb_id.upper()
    query = """
    {
      entry(entry_id: "%s") {
        polymer_entities {
          rcsb_entity_source_organism {
            ncbi_scientific_name
          }
          uniprots {
            rcsb_id
          }
          entity_poly {
            rcsb_entity_polymer_type
          }
        }
      }
    }
    """ % pdb_id

    try:
        data = graphql(query)
    except Exception:
        return []

    entities = (data.get("data", {})
                    .get("entry", {})
                    .get("polymer_entities") or [])

    results = []
    seen = set()  # deduplicate within this PDB

    for entity in entities:
        poly_type = (entity.get("entity_poly") or {}).get("rcsb_entity_polymer_type", "")
        if poly_type != "Protein":
            continue

        organisms = entity.get("rcsb_entity_source_organism") or []
        org_name  = organisms[0].get("ncbi_scientific_name", "Unknown") if organisms else "Unknown"
        uniprots  = entity.get("uniprots") or []

        for u in uniprots:
            uid = u.get("rcsb_id", "")
            if uid and uid not in target_uids and uid not in seen:
                seen.add(uid)
                results.append({
                    "uniprot_id":   uid,
                    "organism":     org_name,
                })

    return results

def _get_gene_name(uniprot_id: str) -> str:
    """Fetch the gene name for a UniProt ID."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    try:
        resp = requests.get(url, params={"format": "json"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        genes = data.get("genes", [])
        if genes:
            return genes[0].get("geneName", {}).get("value", uniprot_id)
    except Exception:
        pass
    return uniprot_id

def _pdbs_for_ligand(ligand_id: str) -> set:
    query = {
        "query": {
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
                "operator": "exact_match",
                "value": ligand_id,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    r = requests.post(SEARCH_URL, json=query, timeout=15)
    if r.status_code != 200:
        return set()
    print({x["identifier"] for x in r.json().get("result_set", [])})
    return {x["identifier"] for x in r.json().get("result_set", [])}


def _pdb_get_resolution(pdb_id: str) -> float:
    """Returns resolution in Angstroms, or 999 if unavailable."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        res = (data.get("refine", [{}])[0].get("ls_d_res_high")
               or data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0])
        return float(res) if res is not None else 999.0
    except Exception:
        return 999.0


def _pick_top5_target_pdbs(all_target_pdbs: list[str]) -> list[tuple]:
    """
    Iterates target PDB IDs, fetching resolution one by one.
    Stops as soon as 5 structures with resolution < 3.0 Å are found.
    If none qualify, returns the 5 with lowest resolution overall.
    Returns list of (pdb_id, resolution) tuples.
    """
    under_3 = []   # (pdb_id, res) with res < 3.0
    all_scored = [] # (pdb_id, res) for fallback

    print(f"  Scanning target PDB IDs for resolution (will stop after 5 under 3.0 Å)...")
    for pdb_id in all_target_pdbs:
        res = _pdb_get_resolution(pdb_id)
        all_scored.append((pdb_id, res))
        print(f"    {pdb_id}: {res:.2f} Å")

        if res < 3.0:
            under_3.append((pdb_id, res))
            if len(under_3) == 5:
                print(f"  Found 5 structures under 3.0 Å — stopping early.")
                break

    if under_3:
        return sorted(under_3, key=lambda x: x[1])
    else:
        print(f"  No structures under 3.0 Å found. Returning 5 with lowest resolution.")
        return sorted(all_scored, key=lambda x: x[1])[:5]



def enrich_with_pdb(
    df: pd.DataFrame,
    human_uniprot: str,
    mouse_uniprot: str,
    chembl_target: dict,         # ← new parameter
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple], dict]:
    """
    Returns:
      enriched_df     — clinical_df with human_pdb_ids, mouse_pdb_ids columns
      pdb_detail_df   — long-form: one row per (molecule, species, ligand_id, pdb_id)
      top5_target_pdbs — list of (pdb_id, resolution) for the target itself
      offtarget_map   — {uniprot_id: {protein_name, ligands: {chembl_id: {pdb_id}}}}

    Reuses PDB API calls: co-bound UniProt IDs are collected during the same
    loop that finds ligand-matched PDBs, avoiding duplicate requests.
    """

    target_uids = {human_uniprot, mouse_uniprot} - {""}

    # ── Extract all UniProt-based PDB IDs from ChEMBL xrefs ──────────────────
    all_target_pdbs = [
        xref["xref_id"]
        for comp in chembl_target.get("target_components", [])
        for xref in comp.get("target_component_xrefs", [])
        if xref.get("xref_src_db") == "PDB"
    ]

    # ── Top 5 target PDBs by resolution (early stopping) ─────────────────────
    print("Selecting top 5 target PDB structures by resolution...")
    top5_target_pdbs = _pick_top5_target_pdbs(all_target_pdbs) if all_target_pdbs else []

    # ── Fetch target PDB sets for ligand matching ─────────────────────────────
    species_map = {}
    for label, uid in [("human", human_uniprot), ("mouse", mouse_uniprot)]:
        print(f"Fetching {label} PDB entries ({uid})...", end=" ", flush=True)
        try:
            pdbs = _uniprot_pdbs(uid)
            print(f"{len(pdbs)} entries")
        except Exception as e:
            print(f"failed ({e})")
            pdbs = set()
        species_map[label] = pdbs

    total = len(df)
    summary     = {}
    detail_rows = []

    # offtarget_map: {uniprot_id: {protein_name, ligands: {chembl_id: set(pdb_ids)}}}
    offtarget_map = defaultdict(lambda: {
        "organism": "Unknown",
        "ligands": defaultdict(set),
    })

    for i, (_, row) in enumerate(df.iterrows(), 1):
        chembl_id = row["molecule_chembl_id"]
        print(f"  [{i}/{total}] {chembl_id}", end=" ... ", flush=True)

        ligand_ids  = _chembl_to_ligand_ids(chembl_id)
        mol_summary = {"human": set(), "mouse": set()}

        # {pdb_id: [co-bound proteins excluding target]}  — built during this loop
        pdb_coproteins: dict[str, list] = {}

        for lid in ligand_ids:
            try:
                ligand_pdbs = _pdbs_for_ligand(lid)
            except Exception:
                continue

            for label, target_pdbs in species_map.items():
                matches = target_pdbs & ligand_pdbs
                for pdb_id in sorted(matches):
                    mol_summary[label].add(pdb_id)
                    detail_rows.append({
                        "molecule_chembl_id": chembl_id,
                        "species":            label,
                        "uniprot_id":         human_uniprot if label == "human" else mouse_uniprot,
                        "ligand_id":          lid,
                        "pdb_id":             pdb_id,
                    })

            # ── Off-target: PDBs where ligand exists but NOT with our target ──────
            all_target_pdbs_set = species_map["human"] | species_map["mouse"]
            offtarget_pdbs = sorted(ligand_pdbs - all_target_pdbs_set)[:15]

            for pdb_id in offtarget_pdbs:
                bound = get_uniprots_for_pdb(pdb_id, target_uids)
                print("bound============",bound)
                for p in bound:
                    
                    uid = p["uniprot_id"]
                    
                    offtarget_map[uid]["organism"] = p["organism"]
                    offtarget_map[uid]["ligands"][chembl_id].add(pdb_id)

            
        # ── No ligand-matched PDBs ────────────────────────────────────────────
        if not mol_summary["human"] and not mol_summary["mouse"]:
            human_pdbs = species_map.get("human", set())
            if human_pdbs:
                print(f"\n  ℹ️  No ligand-matched PDBs for {chembl_id}. "
                      f"Human UniProt PDBs ({human_uniprot}): "
                      f"{', '.join(sorted(human_pdbs))}", end=" ")

        summary[chembl_id] = {
            "human": ", ".join(sorted(mol_summary["human"])),
            "mouse": ", ".join(sorted(mol_summary["mouse"])),
        }
        h = len(mol_summary["human"])
        m = len(mol_summary["mouse"])
        print(f"human={h}  mouse={m}")

    enriched_df = df.copy()
    enriched_df["human_pdb_ids"] = enriched_df["molecule_chembl_id"].map(
        lambda x: summary.get(x, {}).get("human", ""))
    enriched_df["mouse_pdb_ids"] = enriched_df["molecule_chembl_id"].map(
        lambda x: summary.get(x, {}).get("mouse", ""))

    pdb_detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame(
        columns=["molecule_chembl_id", "species", "uniprot_id", "ligand_id", "pdb_id"])

    return enriched_df, pdb_detail_df, top5_target_pdbs, dict(offtarget_map)

def get_chembl_target(uniprot_id: str) -> dict:
    """
    Look up a ChEMBL single-protein target by UniProt accession.
    """
    target = new_client.target
    results = target.filter(
        target_components__accession=uniprot_id,
        target_type="SINGLE PROTEIN"
    )

    hits = list(results)
    if not hits:
        raise ValueError(f"No ChEMBL SINGLE PROTEIN target found for UniProt ID: {uniprot_id}")

    # Display all hits
    print(f"✅ ChEMBL Target(s) found for {uniprot_id}:")
    for h in hits:
        print(f"   {h['target_chembl_id']} — {h['pref_name']} ({h['organism']})")

    # Use the first hit
    chosen = hits[0]
    print(f"\n➡️  Using: {chosen['target_chembl_id']} — {chosen['pref_name']}")
    return chosen

def search_uniprot(query: str, tax_id: int):
    """
    Generic UniProt search helper.
    """

    url = "https://rest.uniprot.org/uniprotkb/search"

    params = {
        "query": query,
        "format": "json",
        "size": 5,
        "fields":
        "accession,protein_name,gene_names,"
        "organism_name,reviewed"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        return None

    top = results[0]

    accession = top["primaryAccession"]

    full_name = (
        top["proteinDescription"]
        ["recommendedName"]
        ["fullName"]
        ["value"]
    )

    gene_name = (
        top.get("genes", [{}])[0]
        .get("geneName", {})
        .get("value", "N/A")
    )

    organism = top["organism"]["scientificName"]

    return {
        "uniprot_id": accession,
        "full_name": full_name,
        "gene": gene_name,
        "organism": organism
    }

def get_uniprot_ids(TARGET_PROTEIN: str) -> dict:
    """
    Workflow:
    1. Find HUMAN entry using protein/gene search
    2. Extract HUMAN gene symbol
    3. Use HUMAN gene symbol to find MOUSE ortholog
    """

    results_dict = {}

    print(f"\nSearching HUMAN UniProt for: {TARGET_PROTEIN}")

    search_url = "https://rest.uniprot.org/uniprotkb/search"

    # ── Try precise field-scoped searches first, fall back to free-text ──
    queries = [
        f'gene_exact:"{TARGET_PROTEIN}" AND organism_id:9606 AND reviewed:true',
        f'{TARGET_PROTEIN} AND organism_id:9606 AND reviewed:true',          # original fallback
    ]

    entry = None
    for query in queries:
        params = {
            "query": query,
            "fields": "accession,protein_name,gene_names,cc_function,cc_disease,"
                      "cc_pathway,cc_tissue_specificity,cc_subunit",
            "format": "json",
            "size": 1,
        }
        resp = requests.get(search_url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            entry = results[0]
            print(f"   ✔ Matched with query: {query}")
            break

    if not entry:
        print("❌ No HUMAN UniProt entry found")
        return {"human": None, "mouse": None}

    # ── Extract fields (unchanged from your original) ──────────────────
    uniprot_id = entry.get("primaryAccession", "")

    protein_desc = entry.get("proteinDescription", {})
    recommended  = protein_desc.get("recommendedName", {})
    full_name    = recommended.get("fullName", {}).get("value", "")

    # Also parse comments so the returned dict mirrors fetch_uniprot ──────
    comments = entry.get("comments", [])
    parsed_comments: dict = {}
    for c in comments:
        ctype = c.get("commentType", "")
        if ctype == "FUNCTION":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            parsed_comments["function"] = " ".join(texts)
        elif ctype == "DISEASE":
            disease = c.get("disease", {})
            parsed_comments.setdefault("diseases", []).append({
                "name":        disease.get("diseaseId", ""),
                "description": disease.get("description", ""),
            })
        elif ctype == "PATHWAY":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            parsed_comments.setdefault("pathways", []).extend(texts)
        elif ctype == "TISSUE SPECIFICITY":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            parsed_comments["tissue_specificity"] = " ".join(texts)
        elif ctype == "SUBUNIT":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            parsed_comments["subunit"] = " ".join(texts)

    genes     = entry.get("genes", [])
    gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""

    human = {
        "uniprot_id": uniprot_id,
        "full_name":  full_name,
        "gene":       gene_name,
        "comments":   parsed_comments,     # ← now consistent with fetch_uniprot
    }

    results_dict["human"] = human

    print(f"\n✅ HUMAN MATCH")
    print(f"   Protein : {human['full_name']}")
    print(f"   Gene    : {human['gene']}")
    print(f"   UniProt : {human['uniprot_id']}")

    # ── Step 2: Mouse ortholog (unchanged) ─────────────────────────────
    human_gene = human["gene"]
    print(f"\nSearching MOUSE using human gene symbol: {human_gene}")

    mouse = search_uniprot(
        query=(
            f'gene_exact:"{human_gene}" '
            f'AND organism_id:10090 '
            f'AND reviewed:true'
        ),
        tax_id=10090
    )

    if not mouse:
        print(f"❌ No mouse UniProt entry found for gene {human_gene}")
        results_dict["mouse"] = None
    else:
        print(f"\n✅ MOUSE MATCH")
        print(f"   Protein : {mouse['full_name']}")
        print(f"   Gene    : {mouse['gene']}")
        print(f"   UniProt : {mouse['uniprot_id']}")
        results_dict["mouse"] = mouse

    return results_dict
    
from chembl_webresource_client.new_client import new_client

def get_clinical_molecules(target_chembl_id: str) -> pd.DataFrame:

    
    molecule  = new_client.molecule
    mechanism = new_client.mechanism
    mec_records = list(mechanism.filter(target_chembl_id=target_chembl_id))
    if not mec_records:
        return pd.DataFrame()
    drug_chembl_ids = list({r["molecule_chembl_id"] for r in mec_records if r.get("molecule_chembl_id")})
    mec_lookup = {
        r["molecule_chembl_id"]: {
            "mechanism_of_action": r.get("mechanism_of_action"),
            "action_type":         r.get("action_type"),
        }
        for r in mec_records if r.get("molecule_chembl_id")
    }
    mol_records = []
    for i in range(0, len(drug_chembl_ids), 200):
        mol_records.extend(list(molecule.filter(molecule_chembl_id__in=drug_chembl_ids[i:i+200])))
    rows = []
    for r in mol_records:
        structures = r.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles") or structures.get("molfile") or None
        rows.append({
            "molecule_chembl_id": r.get("molecule_chembl_id"),
            "pref_name":          r.get("pref_name"),
            "canonical_smiles":   smiles,
            "molecule_type":      r.get("molecule_type"),
            "max_phase":          r.get("max_phase"),
        })
    df = pd.DataFrame(rows)
    df["max_phase"] = pd.to_numeric(df["max_phase"], errors="coerce")
    df["clinical_status"] = df["max_phase"].map({1:"Phase 1", 2:"Phase 2", 3:"Phase 3", 4:"Approved"})
    df["mechanism_of_action"] = df["molecule_chembl_id"].map(lambda x: mec_lookup.get(x, {}).get("mechanism_of_action"))
    df["action_type"]         = df["molecule_chembl_id"].map(lambda x: mec_lookup.get(x, {}).get("action_type"))
    return df










import tkinter as tk
from tkinter import ttk, messagebox
import io
import webbrowser
from PIL import Image, ImageTk
from rdkit.Chem import Draw

def show_selector(TARGET_PROTEIN,df: pd.DataFrame,
                  pdb_detail_df: pd.DataFrame,
                  img_size: int = 180,
                  cols: int = 4):
    """
    Opens a Tk window to select molecules.
    Returns (selected_clinical_df, selected_pdb_detail_df).
    """
    # Destroy any leftover Tk root from a previous run
    try:
        import tkinter as _tk
        for widget in _tk._default_root.winfo_children() if _tk._default_root else []:
            widget.destroy()
        if _tk._default_root:
            _tk._default_root.destroy()
    except Exception:
        pass

    root = tk.Tk()

    root.title("Select Molecules")
    root.geometry("1150x780")
    root.configure(bg="#f5f5f2")

    # Header
    header = tk.Frame(root, bg="#f5f5f2", pady=10)
    header.pack(fill="x", padx=20)
    tk.Label(header, text="Select molecules to save",
             font=("Helvetica", 16, "bold"), bg="#f5f5f2", fg="#1a1a1a").pack(side="left")
    count_var = tk.StringVar(value="0 selected")
    tk.Label(header, textvariable=count_var, font=("Helvetica", 12),
             bg="#f5f5f2", fg="#666").pack(side="left", padx=20)

    # Guide bar
    guide = tk.Frame(root, bg="#EEF2FF", pady=6, padx=16)
    guide.pack(fill="x", padx=20, pady=(0, 6))
    guide_text = (
        "pC (pChEMBL) = −log₁₀(molar potency against primary target)  • |    "
        "Selectivity Δ = on-target pC − best off-target pC  •  "
        "Positive = selective  •  Δ ≥ 2 = 100× selective  •  Δ ≥ 1 = 10× selective"
    )
    tk.Label(guide, text=guide_text, font=("Helvetica", 8), bg="#EEF2FF",
             fg="#3730A3", wraplength=1100, justify="left").pack(anchor="w")

    tk.Label(header, textvariable=count_var, font=("Helvetica", 12),
             bg="#f5f5f2", fg="#666").pack(side="left", padx=20)

    # Scrollable canvas
    frame_outer = tk.Frame(root, bg="#f5f5f2")
    frame_outer.pack(fill="both", expand=True, padx=20)
    canvas = tk.Canvas(frame_outer, bg="#f5f5f2", highlightthickness=0)
    scrollbar = ttk.Scrollbar(frame_outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas, bg="#f5f5f2")
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

    check_vars = []
    images = []

    def update_count(*_):
        count_var.set(f"{sum(v.get() for v in check_vars)} selected")

    phase_colors = {
        4: ("#EAF3DE", "#27500A"),
        3: ("#E6F1FB", "#0C447C"),
        2: ("#FAEEDA", "#633806"),
        1: ("#FAECE7", "#4A1B0C"),
    }

    def pdb_line(label, ids_str, color):
        """Return a formatted label widget for PDB IDs."""
        if ids_str:
            return f"🧬 {label}: {ids_str}"
        return f"🧬 {label}: —"

    for idx, row in df.iterrows():
        r, c = divmod(idx, cols)
        smiles = row.get("canonical_smiles")

        card = tk.Frame(inner, bg="white", bd=1, relief="solid", padx=8, pady=8)
        card.grid(row=r, column=c, padx=8, pady=8, sticky="n")

        # 2D structure
        mol = (
            Chem.MolFromSmiles(smiles)
            if isinstance(smiles, str) and smiles.strip()
            else None
        )
        pil_img = (Draw.MolToImage(mol, size=(img_size, img_size))
                   if mol else Image.new("RGB", (img_size, img_size), "#eeeeee"))
        tk_img  = ImageTk.PhotoImage(pil_img, master=root)
        images.append(tk_img)
        label = tk.Label(card, image=tk_img, bg="white")
        label.image = tk_img   # keep reference to prevent GC
        label.pack()

        # Name
        name = str(row.get("pref_name") or row.get("molecule_chembl_id") or "Unknown")
        tk.Label(card, text=name[:28], font=("Helvetica", 10, "bold"),
                 bg="white", fg="#1a1a1a", wraplength=img_size).pack(pady=(4, 0))

        # Phase badge
        phase  = row.get("max_phase")
        status = f"Phase {int(phase)}" if pd.notna(phase) else "—"
        bg_c, fg_c = phase_colors.get(int(phase) if pd.notna(phase) else 0, ("#eee", "#333"))
        tk.Label(card, text=status, font=("Helvetica", 9),
                bg=bg_c, fg=fg_c, padx=6, pady=2).pack(pady=3)

        # pChEMBL · action
        # pChEMBL · selectivity
        pchembl   = row.get("on_target_best_pchembl")
        sel_delta = row.get("selectivity_delta")
        is_sel    = row.get("is_selective")

        pchembl_str = f"pC={pchembl:.2f}" if pd.notna(pchembl) and pchembl is not None else "pC=—"

        if sel_delta is not None and pd.notna(sel_delta):
            sel_str   = f"Δ={sel_delta:+.2f}"
            sel_color = "#27500A" if is_sel else "#8B0000"
        else:
            sel_str   = "Δ=—"
            sel_color = "#888"

        metric_frame = tk.Frame(card, bg="white")
        metric_frame.pack()
        tk.Label(metric_frame, text=pchembl_str,
                 font=("Helvetica", 8), bg="white", fg="#444").pack(side="left", padx=(0, 6))
        tk.Label(metric_frame, text=sel_str,
                 font=("Helvetica", 8, "bold"), bg="white", fg=sel_color).pack(side="left")
        

        # Separator
        tk.Frame(card, bg="#eeeeee", height=1).pack(fill="x", pady=4)

        # Human PDB IDs
        human_ids = str(row.get("human_pdb_ids") or "").strip()
        tk.Label(card,
                 text=f"Human: {human_ids if human_ids else '—'}",
                 font=("Helvetica", 8, "bold"), bg="white",
                 fg="#1a6bb5" if human_ids else "#bbb",
                 wraplength=img_size, justify="left").pack(anchor="w")

        # Mouse PDB IDs
        mouse_ids = str(row.get("mouse_pdb_ids") or "").strip()
        tk.Label(card,
                 text=f"Mouse:  {mouse_ids if mouse_ids else '—'}",
                 font=("Helvetica", 8, "bold"), bg="white",
                 fg="#2e7d32" if mouse_ids else "#bbb",
                 wraplength=img_size, justify="left").pack(anchor="w")

        # Checkbox
        var = tk.BooleanVar()
        var.trace_add("write", update_count)
        check_vars.append(var)
        tk.Checkbutton(card, text="Select", variable=var, bg="white",
                       font=("Helvetica", 9), fg="#333",
                       activebackground="white").pack(pady=(6, 0))

    # Buttons
    btn_frame = tk.Frame(root, bg="#f5f5f2", pady=12)
    btn_frame.pack(fill="x", padx=20)
    result_holder = [None, None]

    def save_selected():
        idxs = [i for i, v in enumerate(check_vars) if v.get()]
        if not idxs:
            messagebox.showwarning("Nothing selected", "Please select at least one molecule.")
            return

        selected_chembl_ids = set(df.iloc[idxs]["molecule_chembl_id"])

        # Full clinical_df rows for selected molecules
        sel_clinical = df.iloc[idxs].reset_index(drop=True)

        # All pdb_detail rows for selected molecules
        sel_pdb = (pdb_detail_df[pdb_detail_df["molecule_chembl_id"].isin(selected_chembl_ids)]
                   .reset_index(drop=True))

        result_holder[0] = sel_clinical
        result_holder[1] = sel_pdb

        import os

        os.makedirs(TARGET_PROTEIN, exist_ok=True)

        out_file = f"{TARGET_PROTEIN}/csvs/{TARGET_PROTEIN}_selected_ref_molecules.csv"
        print("sel_clinical\n",sel_clinical)
        print("sel_pdb\n",sel_pdb)

        # Merge molecule info with pdb/ligand detail rows
        if sel_pdb.empty:
            final_df = sel_clinical

        else:

            final_df = sel_pdb.merge(
                sel_clinical,
                on="molecule_chembl_id",
                how="left"
            )
        print(final_df)

        # Save final dataframe
        final_df.to_csv(out_file, index=False)
        # sel_pdb.to_csv("selected_pdb_details.csv", index=False)

        print(f"\n✅ Saved {len(idxs)} molecules to selected_molecules.csv")
        # print(f"✅ Saved {len(sel_pdb)} PDB rows to selected_pdb_details.csv")
        print("\n── selected_molecules.csv ──")
        print(sel_clinical.to_string(index=False))
        # print("\n── selected_pdb_details.csv ──")
        # print(sel_pdb.to_string(index=False))

        messagebox.showinfo(
            "Saved",
            f"{len(idxs)} molecules → selected_molecules.csv\n"
            # f"{len(sel_pdb)} PDB rows → selected_pdb_details.csv"
        )
        root.destroy()

    style = {"font": ("Helvetica", 11), "padx": 16, "pady": 6, "bd": 0, "cursor": "hand2"}
    tk.Button(btn_frame, text="Select all", bg="#e8e8e4", fg="#1a1a1a",
              command=lambda: [v.set(True) for v in check_vars], **style).pack(side="left", padx=4)
    tk.Button(btn_frame, text="Clear", bg="#e8e8e4", fg="#1a1a1a",
              command=lambda: [v.set(False) for v in check_vars], **style).pack(side="left", padx=4)
    tk.Button(btn_frame, text="💾  Save selected", bg="#1a1a1a", fg="white",
              command=save_selected, **style).pack(side="right", padx=4)

    root.mainloop()
    return result_holder[0], result_holder[1]


def save_ids_and_molecules(
    target_protein: str,
    uniprot_info: dict,
    chembl_target: dict,
    clinical_df: pd.DataFrame,
    pdb_detail_df: pd.DataFrame,
    top5_target_pdbs: list[tuple],     # ← new
    offtarget_map: dict,               # ← new
):
    lines = []

    # ── IDs block ─────────────────────────────────────────────────────────────
    lines.append(f"TARGET: {target_protein}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("IDENTIFIERS")
    lines.append("-" * 40)
    human = uniprot_info.get("human", {})
    mouse = uniprot_info.get("mouse", {})
    lines.append(f"Human UniProt ID : {human.get('uniprot_id', 'N/A')}")
    lines.append(f"Human Gene       : {human.get('gene', 'N/A')}")
    lines.append(f"Human Full Name  : {human.get('full_name', 'N/A')}")
    lines.append(f"Mouse UniProt ID : {mouse.get('uniprot_id', 'N/A')}")
    lines.append(f"Mouse Gene       : {mouse.get('gene', 'N/A')}")
    lines.append(f"ChEMBL Target ID : {chembl_target.get('target_chembl_id', 'N/A')}")
    lines.append("")

    # ── Top 5 target PDB structures ───────────────────────────────────────────
    lines.append("TARGET PDB STRUCTURES (Top 5 by resolution)")
    lines.append("-" * 40)
    if top5_target_pdbs:
        for pdb_id, res in top5_target_pdbs:
            res_str = f"{res:.2f} Å" if res < 999 else "N/A"
            lines.append(f"  {pdb_id}  (resolution: {res_str})")
    else:
        lines.append("  None found.")
    lines.append("")

    # ── Molecule table ────────────────────────────────────────────────────────
    lines.append("CLINICAL MOLECULES")
    lines.append("-" * 40)

    if clinical_df is not None and not clinical_df.empty:
        cols = [
            "molecule_chembl_id", "pref_name",
            "max_phase", "on_target_best_pchembl", "selectivity_delta",
        ]
        widths = {
            "molecule_chembl_id":  18,
            "pref_name":           20,
            "max_phase":           10,
            "on_target_best_pchembl": 10,
            "selectivity_delta": 10,
        }
        headers = {
            "molecule_chembl_id":  "ChEMBL ID",
            "pref_name":           "Name",
            "max_phase":           "Max Phase",
            "on_target_best_pchembl": "On-Target Best pChemBL",
            "selectivity_delta":   "Off target Selectivity Delta",
        }

        header_row = "  ".join(headers[c].ljust(widths[c]) for c in cols)
        separator  = "  ".join("-" * widths[c] for c in cols)
        lines.append(header_row)
        lines.append(separator)

        for _, row in clinical_df.iterrows():
            data_row = "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
            lines.append(data_row)
        lines.append("")

        # PDB structures per molecule
        if pdb_detail_df is not None and not pdb_detail_df.empty:
            lines.append("PDB STRUCTURES PER MOLECULE (ligand-matched)")
            lines.append("-" * 40)
            for chembl_id, group in pdb_detail_df.groupby("molecule_chembl_id"):
                for species, sgroup in group.groupby("species"):
                    pdb_ids = ", ".join(sorted(sgroup["pdb_id"].unique()))
                    lines.append(f"  {chembl_id} ({species}): {pdb_ids}")
            lines.append("")
    else:
        lines.append("  No clinical molecules found.")
        lines.append("")

    # ── Off-target analysis ───────────────────────────────────────────────────
    lines.append("OFF-TARGET PROTEINS BOUND BY CLINICAL MOLECULES")
    lines.append("-" * 40)
    if not offtarget_map:
        lines.append("  No off-target proteins identified.")
    else:
        # Resolve gene names for all off-target UniProt IDs
        print("Fetching gene names for off-target proteins...")
        gene_names = {uid: _get_gene_name(uid) for uid in offtarget_map}

        # Sort by number of ligands (descending)
        sorted_offtargets = sorted(
            offtarget_map.items(),
            key=lambda x: len(x[1]["ligands"]),
            reverse=True
        )

        for uid, info in sorted_offtargets:
            gene      = gene_names.get(uid, uid)
            organism  = info.get("organism", "Unknown")
            ligands   = info["ligands"]

          
            lines.append(f" Protein:  {uid}  |  {gene}  |  {organism}")
            lines.append(f"  Bound by {len(ligands)} ligand(s):")
            for chembl_id, pdb_ids in sorted(ligands.items()):
                rep_pdb = sorted(pdb_ids)[0]
                lines.append(f"    - {chembl_id}: {rep_pdb}")
            lines.append("")

    # ── Write file ────────────────────────────────────────────────────────────
    out_path = f"{target_protein}/txts/{target_protein}_ids_and_molecules.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {out_path}")
    return out_path