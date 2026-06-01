# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
import os
import re
import csv
import time
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from datetime import datetime
from collections import Counter, defaultdict


EPO_MAX_RESULTS    = 100
EPO_BATCH_SIZE     = 25
EPO_DELAY_SECONDS  = 1.0

CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════════
BIOLOGICAL_SUFFIXES = (
    "mab","umab","ximab","zumab","mumab","tuzumab","cept","afacept","anercept",
    "kin","fermin","tropin","poietin","ase","feron","factor","gene","cell",
    "vector","viral","vac","rix","vax","tide","peptide","mersen","mirsen","sen"
)
SMALL_MOLECULE_SUFFIXES = (
    "nib","previr","statin","olol","pril","sartan","azole","conazole","mycin",
    "cillin","cycline","floxacin","mustine","platin","parib","ciclib","limus",
    "vir","uvir","thiazide","semide","zepam","zodone"
)
BIOLOGICAL_KEYWORDS = {
    "antibody","antibodies","monoclonal","biologic","biological","protein",
    "enzyme","hormone","cytokine","interleukin","interferon","vaccine","toxin",
    "serum","plasma","stem cell","car-t","car t","t-cell","nk cell",
    "gene therapy","mrna","sirna","antisense","oligonucleotide","peptide",
    "recombinant","fusion protein","growth factor","erythropoietin","insulin",
    "glucagon"
}
SMALL_MOLECULE_KEYWORDS = {
    "tablet","capsule","oral","pill","inhibitor","agonist","antagonist",
    "hydrochloride","phosphate","sulfate","mesylate","acetate","citrate",
    "tartrate","small molecule"
}
DEVICE_KEYWORDS = {
    "device","implant","catheter","stent","pump","monitor","sensor",
    "electrode","prosthesis","scaffold","patch","ventilator","pacemaker",
    "defibrillator","laser"
}
PROCEDURE_KEYWORDS = {
    "surgery","surgical","resection","transplant","biopsy","infusion",
    "injection","radiation","radiotherapy","chemotherapy","ablation","bypass",
    "catheterization","endoscopy","dialysis","physiotherapy","acupuncture"
}
BEHAVIORAL_KEYWORDS = {
    "counseling","counselling","cognitive","behavioral","behaviour",
    "psychotherapy","mindfulness","meditation","education","training",
    "coaching","support group","lifestyle","physical activity","smoking cessation"
}
SUPPLEMENT_KEYWORDS = {
    "vitamin","mineral","supplement","herbal","nutraceutical","probiotic",
    "prebiotic","omega","calcium","zinc","magnesium","folate","folic acid"
}
GENETIC_KEYWORDS = {
    "crispr","cas9","plasmid","viral vector","lentiviral","adeno-associated",
    "aav","retroviral","transfection","transgene","genome editing","knockdown",
    "knockout","rna interference"
}
RADIATION_KEYWORDS = {
    "radiation","radiotherapy","proton therapy","gamma","brachytherapy",
    "radiosurgery","stereotactic","cyberknife"
}
DIAGNOSTIC_KEYWORDS = {
    "imaging","scan","mri","ct scan","pet scan","ultrasound","x-ray","biopsy",
    "blood test","biomarker","assay","diagnostic","test kit","sequencing","pcr",
    "elisa","flow cytometry","immunohistochemistry","western blot","ecg","eeg",
    "spirometry"
}

CHART_COLORS = [
    "#D97706",  # amber
    "#A8A29E",  # stone grey
    "#78716C",  # warm grey
    "#525252",  # dark grey
    "#1F1F1F",  # near black
    "#C2410C",  # deep orange
]


def classify_intervention(name, ctgov_type):
    n  = name.lower().strip()
    ct = (ctgov_type or "").upper().strip()
    if ct == "DEVICE":              return "DEVICE"
    if ct == "PROCEDURE":           return "PROCEDURE"
    if ct == "BEHAVIORAL":          return "BEHAVIORAL"
    if ct == "DIETARY SUPPLEMENT":  return "DIETARY SUPPLEMENT"
    if ct == "GENETIC":             return "GENETIC / GENE THERAPY"
    if ct == "RADIATION":           return "RADIATION"
    if ct == "COMBINATION PRODUCT": return "COMBINATION PRODUCT"
    if ct == "DIAGNOSTIC TEST":     return "DIAGNOSTIC TEST"
    if any(kw in n for kw in DIAGNOSTIC_KEYWORDS):  return "DIAGNOSTIC TEST"
    if any(kw in n for kw in GENETIC_KEYWORDS):     return "GENETIC / GENE THERAPY"
    if any(kw in n for kw in RADIATION_KEYWORDS):   return "RADIATION"
    if any(kw in n for kw in DEVICE_KEYWORDS):      return "DEVICE"
    if any(kw in n for kw in PROCEDURE_KEYWORDS):   return "PROCEDURE"
    if any(kw in n for kw in BEHAVIORAL_KEYWORDS):  return "BEHAVIORAL"
    if any(kw in n for kw in SUPPLEMENT_KEYWORDS):  return "DIETARY SUPPLEMENT"
    if any(kw in n for kw in BIOLOGICAL_KEYWORDS):  return "BIOLOGICAL"
    if any(n.rstrip(" ).,").endswith(s) for s in BIOLOGICAL_SUFFIXES):  return "BIOLOGICAL"
    if any(kw in n for kw in SMALL_MOLECULE_KEYWORDS): return "SMALL MOLECULE"
    if any(n.rstrip(" ).,").endswith(s) for s in SMALL_MOLECULE_SUFFIXES): return "SMALL MOLECULE"
    if ct in ("DRUG", "BIOLOGICAL"): return f"{ct} (UNCLASSIFIED)"
    return ct if ct else "OTHER"


def is_codename(name):
    n = name.strip()
    if not (re.search(r'[A-Za-z]', n) and re.search(r'\d', n)):
        return False
    for tok in re.split(r'[\s\-/]+', n):
        if re.search(r'[A-Za-z]', tok) and re.search(r'\d', tok):
            return True
    return False


def parse_date(ds):
    if not ds:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(ds, fmt)
        except ValueError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_all_studies(query_term):
    """
    Fetch all studies from ClinicalTrials.gov for a single query term.
    Returns a list of raw study dicts, or [] on any error.
    """
    studies, params = [], {
        "query.term": query_term,
        "fields": (
            "NCTId,BriefTitle,OverallStatus,StartDate,CompletionDate,"
            "Condition,LocationCountry,WhyStopped,Phase,StudyType,"
            "LeadSponsorName,CollaboratorName,InterventionName,InterventionType"
        ),
        "pageSize": 100,
        "format": "json",
    }
    next_token = None
    while True:
        if next_token:
            params["pageToken"] = next_token
        try:
            r = requests.get(CTGOV_BASE, params=params, timeout=30)
            r.raise_for_status()
        except Exception:
            break
        d = r.json()
        studies.extend(d.get("studies", []))
        next_token = d.get("nextPageToken")
        if not next_token:
            break
    return studies


def fetch_and_merge_studies(protein_name, df_selective=None):
    """
    Fetch studies for protein_name, and optionally for every entry in the
    'Name' column of df_selective.  Deduplicates on NCTId before returning.
    """
    print(f"  [{protein_name}] fetching…", end=" ")
    all_studies = fetch_all_studies(protein_name)
    print(f"{len(all_studies)} studies")

    if df_selective is not None and not df_selective.empty and "pref_name" in df_selective.columns:
        extra_terms = [
            str(n).strip()
            for n in df_selective["pref_name"].dropna().unique()
            if str(n).strip()
        ]
        for term in extra_terms:
            print(f"  [{term}] fetching…", end=" ")
            try:
                batch = fetch_all_studies(term)
                print(f"{len(batch)} studies")
                all_studies.extend(batch)
            except Exception:
                print("skipped")

    seen   = set()
    unique = []
    for s in all_studies:
        nct_id = (
            s.get("protocolSection", {})
             .get("identificationModule", {})
             .get("nctId", "")
        )
        if nct_id and nct_id not in seen:
            seen.add(nct_id)
            unique.append(s)
        elif not nct_id:
            unique.append(s)

    print(f"  Total unique studies after deduplication: {len(unique)}")
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
# LLM INTERVENTION FILTER
# ═══════════════════════════════════════════════════════════════════════════════
_FILTER_PROMPT = """You are a drug discovery expert. Below is a numbered list of intervention names
from clinical trial records. Your task is to identify which ones are genuine
clinical candidates — meaning specifically named drugs, biologics, investigational
compounds, or therapeutic agents (including alphanumeric code-names like ABC-123,
monoclonal antibodies, small molecules, gene therapies, vaccines, approved drugs, etc.).
 
EXCLUDE — do NOT include any entry that is:
- A placebo or sham, in any form or capitalisation
  (e.g. "Placebo", "PLACEBO", "Matching Placebo", "Placebo for X", "placebo")
- A study arm label or reference treatment descriptor
  (e.g. "Reference Treatment", "Standard of Care", "SOC", "Active Comparator")
- A generic class descriptor rather than a specific named agent
  (e.g. "Tyk2 inhibitor", "anti-CD20 antibody", "checkpoint inhibitor")
- A modifier or qualifier of another drug name
  (e.g. "X Matching Placebo", "X Reference", "X Dose")
- A procedure, surgery, device, behavioral intervention, or diagnostic test
- A dietary supplement or nutrient (unless it is a defined investigational agent)
- A dose description, free-text entry, or general descriptor
 
INCLUDE — only entries that are:
- A specific proprietary name, INN, or alphanumeric code-name of a drug or biologic
- Clearly identifiable as a therapeutic agent in its own right
 
Return ONLY a JSON array of the integer numbers corresponding to entries that
ARE clinical candidates. No explanation, no markdown, no extra text.
Example output: [1, 3, 5]
 
Intervention list:
{entries}
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5:7b-instruct"


def _llm_filter_batch(batch):
    """
    Send one batch (list of str) to Qwen via Ollama.
    Returns the subset of entries the LLM considers clinical candidates.
    On any failure, returns the full batch unchanged.
    """
    numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(batch))
    prompt   = _FILTER_PROMPT.format(entries=numbered)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": QWEN_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        import json as _json
        raw_clean = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
        indices   = _json.loads(raw_clean)

        if not isinstance(indices, list):
            raise ValueError("LLM did not return a list")

        kept = [batch[i - 1] for i in indices
                if isinstance(i, int) and 1 <= i <= len(batch)]
        return kept

    except Exception as e:
        print(f"    [LLM filter] batch failed ({e}), keeping all {len(batch)} entries")
        return batch


def _llm_filter_interventions(study_interventions_map, batch_size=50):
    """
    Filter intervention names through the LLM, then split results into
    standalone interventions and combination products based on per-study
    survivor counts.  Deduplication (case-insensitive) is applied AFTER
    the LLM so that the same drug spelled differently in two studies is
    not mistaken for a combination.

    Parameters
    ----------
    study_interventions_map : dict
        {nct_id: [iv_name, ...]}  — noise-filtered names per study,
        in the order they were collected.

    Returns
    -------
    interventions_all : list of (canonical_name, nct_ids_str)
        Studies where exactly one intervention survived the LLM filter.
        Sorted alphabetically by canonical name.

    combo_rows : list of (interventions_str, nct_id)
        Studies where two or more interventions survived the LLM filter.
        interventions_str is the survivor names joined by " + ".
        Sorted by nct_id.
    """
    if not study_interventions_map:
        return [], []

    # ── Step 1: collect all unique names across all studies (pre-dedup) ──────
    all_names_ordered = []
    seen_for_dedup    = set()
    for names in study_interventions_map.values():
        for n in names:
            if n not in seen_for_dedup:
                seen_for_dedup.add(n)
                all_names_ordered.append(n)

    print(f"  Unique intervention names before LLM: {len(all_names_ordered)}")
    print(f"  LLM filtering in batches of {batch_size}…")

    # ── Step 2: LLM filter on the raw (pre-dedup) unique name list ───────────
    kept_names_raw = []
    for start in range(0, len(all_names_ordered), batch_size):
        batch  = all_names_ordered[start: start + batch_size]
        result = _llm_filter_batch(batch)
        kept_names_raw.extend(result)
        print(f"    batch {start // batch_size + 1}: "
              f"{len(batch)} in → {len(result)} kept")

    kept_set_raw = set(kept_names_raw)
    print(f"  LLM filter complete: {len(kept_set_raw)}/{len(all_names_ordered)} retained")

    # ── Step 3: case-insensitive dedup AFTER LLM ─────────────────────────────
    # Build canonical name map: lowercase → canonical form
    seen_lower    = {}   # lowercase → canonical name
    canonical_map = {}   # original name → canonical name
    for name in kept_names_raw:
        key = name.lower()
        if key not in seen_lower:
            seen_lower[key] = name
            canonical_map[name] = name
        else:
            canonical = seen_lower[key]
            # Prefer more-uppercase form, then longer, as canonical
            if (sum(c.isupper() for c in name), len(name)) > \
               (sum(c.isupper() for c in canonical), len(canonical)):
                seen_lower[key] = name
                # Remap old canonical → new canonical
                for k, v in canonical_map.items():
                    if v == canonical:
                        canonical_map[k] = name
            canonical_map[name] = seen_lower[key]

    # ── Step 4: per-study, collect surviving canonical names ─────────────────
    # standalone_map : canonical_name → set of nct_ids (single-survivor studies)
    # combo_rows     : list of (sorted_canonical_names_tuple, nct_id)
    standalone_map = defaultdict(set)   # canonical → {nct_id, ...}
    combo_rows     = []                 # [(interventions_str, nct_id), ...]

    for nct_id, names in study_interventions_map.items():
        # Map each name through canonical_map; skip names the LLM dropped
        survivors = []
        seen_canonical = set()
        for n in names:
            if n in kept_set_raw:
                canon = canonical_map.get(n, n)
                if canon not in seen_canonical:
                    seen_canonical.add(canon)
                    survivors.append(canon)

        if len(survivors) == 1:
            standalone_map[survivors[0]].add(nct_id)
        elif len(survivors) >= 2:
            combo_rows.append((" + ".join(sorted(survivors)), nct_id))

    # ── Step 5: build final output lists ─────────────────────────────────────
    interventions_all = [
        (name, ", ".join(sorted(nct_ids)))
        for name, nct_ids in sorted(standalone_map.items(), key=lambda x: x[0].lower())
    ]

    combo_rows_sorted = sorted(combo_rows, key=lambda x: x[1])

    return interventions_all, combo_rows_sorted


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_studies(studies, protein_name):
    """
    Process raw ClinicalTrials.gov studies into structured data for both
    the text/chart report and the sponsor/collaborator CSV.
    """
    now_year    = datetime.now().year
    cutoff_5yr  = now_year - 5
    cutoff_2yr  = now_year - 2
    cutoff_10yr = now_year - 10

    status_counter = Counter()
    conditions     = Counter()
    sponsors       = Counter()
    collaborators  = Counter()
    withdrawn_info = []

    # nct_id → [iv_name, ...] — noise-filtered intervention names per study
    # (ordered list so we preserve per-study grouping for combo detection)
    study_interventions_map = defaultdict(list)

    year_counts = Counter()

    sponsor_year_count  = defaultdict(lambda: defaultdict(int))
    collab_year_count   = defaultdict(lambda: defaultdict(int))
    sponsor_studies     = defaultdict(list)
    collab_studies      = defaultdict(list)

    sponsor_meta      = {}
    collaborator_meta = {}

    def update_meta(meta_dict, counter, key, year, nct_id, title, interventions):
        counter[key] += 1
        if key not in meta_dict or year > meta_dict[key][0]:
            meta_dict[key] = (year, nct_id, title, interventions)

    for s in studies:
        proto  = s.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        st_mod = proto.get("statusModule", {})
        co_mod = proto.get("conditionsModule", {})
        lo_mod = proto.get("contactsLocationsModule", {})
        sp_mod = proto.get("sponsorCollaboratorsModule", {})
        ar_mod = proto.get("armsInterventionsModule", {})
        de_mod = proto.get("designModule", {})

        nct_id = id_mod.get("nctId", "N/A")
        title  = id_mod.get("briefTitle", "N/A")
        status = st_mod.get("overallStatus", "UNKNOWN")
        why    = st_mod.get("whyStopped", "")
        start  = st_mod.get("startDateStruct", {}).get("date", "")
        end    = st_mod.get("completionDateStruct", {}).get("date", "")
        phase  = " / ".join(de_mod.get("phases", [])) or "N/A"

        status_counter[status] += 1
        if status == "WITHDRAWN":
            withdrawn_info.append({
                "title": title, "nct_id": nct_id,
                "why": why or "Reason not stated", "start": start,
            })

        for cond in co_mod.get("conditions", []):
            conditions[cond] += 1

        locs = list({
            l.get("country", "")
            for l in lo_mod.get("locations", [])
            if l.get("country", "")
        })

        lead   = sp_mod.get("leadSponsor", {}).get("name", "")
        colabs = [c.get("name", "") for c in sp_mod.get("collaborators", []) if c.get("name")]
        if lead:
            sponsors[lead] += 1
        for col in colabs:
            collaborators[col] += 1

        dt         = parse_date(start)
        study_year = dt.year if dt else None
        if study_year:
            year_counts[study_year] += 1

        study_stub = {
            "title": title, "nct_id": nct_id, "phase": phase,
            "status": status, "start": start, "end": end,
        }

        if study_year and study_year >= cutoff_10yr:
            if lead:
                sponsor_year_count[lead][study_year] += 1
            for col in colabs:
                collab_year_count[col][study_year] += 1
            if lead:
                sponsor_studies[lead].append(study_stub)
            for col in colabs:
                collab_studies[col].append(study_stub)

        if not (study_year and study_year >= cutoff_5yr):
            continue

        iv_names_for_csv = []
        for iv in ar_mod.get("interventions", []):
            iv_name = iv.get("name", "").strip()
            if not iv_name:
                continue

            _CS_EXCLUDE = {
                "mg", "Dose", "blood", "method", "capsule", "tablet", "text",
                "SOC", "Reference Treatment", "Standard of Care", "Matching",
            }
            _CI_EXCLUDE = {"placebo", "inhibitor", "antibody", "comparator"}
            iv_lower = iv_name.lower()
            is_noise = (
                any(term in iv_name for term in _CS_EXCLUDE) or
                any(term in iv_lower for term in _CI_EXCLUDE)
            )
            if not is_noise:
                # Collect per-study for LLM filter + combo detection later
                study_interventions_map[nct_id].append(iv_name)
            iv_names_for_csv.append(iv_name)

        if study_year and study_year >= cutoff_5yr and iv_names_for_csv:
            iv_str = "; ".join(iv_names_for_csv)
            if lead:
                update_meta(sponsor_meta, Counter(), lead, study_year, nct_id, title, iv_str)
            for col in colabs:
                update_meta(collaborator_meta, Counter(), col, study_year, nct_id, title, iv_str)

    csv_sponsor_counter      = Counter()
    csv_collaborator_counter = Counter()
    for s in studies:
        proto  = s.get("protocolSection", {})
        st_mod = proto.get("statusModule", {})
        sp_mod = proto.get("sponsorCollaboratorsModule", {})
        ar_mod = proto.get("armsInterventionsModule", {})
        start  = st_mod.get("startDateStruct", {}).get("date", "")
        dt     = parse_date(start)
        study_year = dt.year if dt else None
        if not (study_year and study_year >= cutoff_5yr):
            continue
        iv_names = [iv.get("name", "") for iv in ar_mod.get("interventions", []) if iv.get("name", "")]
        if not iv_names:
            continue
        lead   = sp_mod.get("leadSponsor", {}).get("name", "")
        colabs = [c.get("name", "") for c in sp_mod.get("collaborators", []) if c.get("name")]
        if lead:
            csv_sponsor_counter[lead] += 1
        for col in colabs:
            csv_collaborator_counter[col] += 1

    total     = len(studies)
    completed = status_counter.get("COMPLETED", 0)
    withdrawn = status_counter.get("WITHDRAWN", 0)
    ongoing   = sum(
        v for k, v in status_counter.items()
        if k in ("RECRUITING", "ACTIVE_NOT_RECRUITING",
                  "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING")
    )

    top_sponsors = [s for s, _ in sponsors.most_common(6)]
    top_collabs  = [c for c, _ in collaborators.most_common(6)]

    # LLM filter — splits into standalone interventions and combo rows
    interventions_all, combo_rows = _llm_filter_interventions(dict(study_interventions_map))

    return {
        "protein_name": protein_name,
        "total": total, "completed": completed,
        "withdrawn_count": withdrawn, "ongoing": ongoing,
        "other": total - completed - withdrawn - ongoing,
        "status_counter": status_counter,
        "conditions": conditions,
        # list of (intervention_name, nct_ids_str) — standalone, sorted alphabetically
        "interventions_all": interventions_all,
        # list of (interventions_str, nct_id) — multi-drug studies, sorted by nct_id
        "combo_rows": combo_rows,
        "withdrawn_info": withdrawn_info,
        "sponsor_year_count": sponsor_year_count,
        "collab_year_count": collab_year_count,
        "sponsor_studies": sponsor_studies,
        "collab_studies": collab_studies,
        "top_sponsors": top_sponsors,
        "top_collabs": top_collabs,
        "year_counts": year_counts,
        "sponsors_full": sponsors,
        "collaborators_full": collaborators,
        "sponsor_meta": sponsor_meta,
        "collaborator_meta": collaborator_meta,
        "csv_sponsor_counter": csv_sponsor_counter,
        "csv_collaborator_counter": csv_collaborator_counter,
        "now_year": now_year,
        "cutoff_10yr": cutoff_10yr,
        "cutoff_5yr": cutoff_5yr,
        "cutoff_2yr": cutoff_2yr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHART SAVERS
# ═══════════════════════════════════════════════════════════════════════════════
def save_trend_chart(data, out_dir):
    yc = data["year_counts"]
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    if not yc:
        ax.text(0.5, 0.5, "No dated study data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#666666")
    else:
        min_y, max_y = min(yc), max(yc)
        years  = list(range(min_y, max_y + 1))
        counts = [yc.get(y, 0) for y in years]

        ax.fill_between(years, counts, alpha=0.30, color="#D6D3D1", zorder=1)
        ax.plot(years, counts, color="#C45A1A", linewidth=2.2,
                marker="o", markersize=5.2, markerfacecolor="#C45A1A",
                markeredgewidth=0, zorder=4)

        if len(years) >= 4:
            deg = min(3, len(years) - 1)
            z   = np.polyfit(years, counts, deg)
            xs  = np.linspace(min_y, max_y, 200)
            ys  = np.maximum(np.poly1d(z)(xs), 0)
            ax.plot(xs, ys, color="#4B5563", linewidth=1.7,
                    linestyle="--", label="Trend", zorder=3)
            ax.legend(fontsize=8, frameon=True,
                      facecolor="#F8F8F8", edgecolor="#CCCCCC")

        for y, c in zip(years, counts):
            if c > 0:
                ax.annotate(str(c), (y, c), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7.5, color="#2F2F2F")

        ax.set_xlim(min_y - 0.5, max_y + 0.5)
        ax.set_ylim(0, max(counts) * 1.35 if counts else 5)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years],
                           rotation=45, fontsize=7.5, color="#4A4A4A")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(axis="y", labelsize=8, colors="#4A4A4A")
        ax.set_ylabel("New Studies", fontsize=9, color="#222222")

    ax.set_title("Clinical Trials Started per Year", fontsize=11,
                 fontweight="bold", color="#111111", loc="left", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.7, linestyle=":", alpha=0.9)
    plt.tight_layout()

    path = os.path.join(out_dir, "chart_trend.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_entity_panel_chart(entity_yc, top, years, title, filename, out_dir):
    n = len(top)
    if n == 0:
        fig, ax = plt.subplots(figsize=(11, 2.5))
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="grey")
        ax.axis("off")
        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(11, rows * 2.6 + 0.4),
                              sharex=False, sharey=False)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, entity in enumerate(top):
        ax  = axes[i]
        cnt = [entity_yc[entity].get(y, 0) for y in years]
        col = CHART_COLORS[i % len(CHART_COLORS)]
        ax.bar(years, cnt, color=col, alpha=0.60, width=0.55, zorder=2)

        ax2 = ax.twinx()
        ax2.plot(years, np.cumsum(cnt), color=col, linewidth=1.8,
                 linestyle="-", marker=".", markersize=4.5, alpha=0.95, zorder=3)
        ax2.tick_params(axis="y", labelsize=6, colors=col)
        ax2.set_ylabel("Cumul.", fontsize=5.5, color=col)
        ax2.spines["right"].set_color(col)
        ax2.spines["right"].set_alpha(0.35)
        ax2.spines["top"].set_visible(False)

        short = (entity[:26] + "…") if len(entity) > 26 else entity
        ax.set_title(short, fontsize=7.5, fontweight="bold", pad=3)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y)[-2:] for y in years], fontsize=6, rotation=45)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_facecolor("#F8FAFB")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E0E6ED", linewidth=0.5, linestyle=":")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=10.5, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.0)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CSV WRITER
# ═══════════════════════════════════════════════════════════════════════════════
def write_trials_csv(data, output_csv):
    """
    Write sponsor/collaborator summary CSV from already-processed study data.
    """
    sponsor_meta      = data["sponsor_meta"]
    collaborator_meta = data["collaborator_meta"]
    csv_sp_counter    = data["csv_sponsor_counter"]
    csv_co_counter    = data["csv_collaborator_counter"]
    print("SPONSOR META",sponsor_meta)
    print("COLLABORATOR META",collaborator_meta)

    rows = []

    for name, (year, nct_id, title, interventions) in sponsor_meta.items():
        rows.append({
            "Entity Type":       "Sponsor",
            "Name":              name,
            "Study Count":       csv_sp_counter[name],
            "Most Recent Year":  year,
            "NCT ID":            nct_id,
            "Study Title":       title,
            "Intervention Name": interventions,
        })

    for name, (year, nct_id, title, interventions) in collaborator_meta.items():
        rows.append({
            "Entity Type":       "Collaborator",
            "Name":              name,
            "Study Count":       csv_co_counter[name],
            "Most Recent Year":  year,
            "NCT ID":            nct_id,
            "Study Title":       title,
            "Intervention Name": interventions,
        })

    rows.sort(key=lambda x: (x["Study Count"], x["Most Recent Year"]), reverse=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Entity Type", "Name", "Study Count", "Most Recent Year",
            "NCT ID", "Study Title", "Intervention Name"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Trials CSV saved → {output_csv}  ({len(rows)} rows)")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════════════
SEP  = "=" * 70
SEP2 = "-" * 70


def _col(items, widths):
    return "  ".join(str(v)[:w].ljust(w) for v, w in zip(items, widths))


def _table(headers, rows, widths):
    lines = [_col(headers, widths), "  ".join("-" * w for w in widths)]
    for row in rows:
        lines.append(_col(row, widths))
    return "\n".join(lines)


def write_txt_report(data, chart_paths, out_path):
    pn  = data["protein_name"]
    now = datetime.now().strftime("%B %d, %Y")
    lines = []

    # ── Cover ────────────────────────────────────────────────────────────────
    lines += [
        SEP,
        "  TARGET LANDSCAPE REPORT",
        f"  Protein / Target : {pn.upper()}",
        f"  Generated        : {now}",
        "  Data Source      : ClinicalTrials.gov",
        SEP, "",
        "CONTENTS", SEP2,
        "  1. Clinical Trials",
        "     1.1  Overview & Trial Trend",
        "     1.2  Associated Conditions",
        "     1.3  All Interventions (unique, from trial data)",
        "     1.4  Combination Therapies / Products",
        "     1.5  Lead Sponsors",
        "     1.6  Collaborating Institutions",
        "     1.7  Withdrawn Studies",
        "",
        "  NOTE ON CHARTS",
        "  Charts are saved as separate .png files in the same output folder.",
        "  Each section references the relevant chart filename.", "",
    ]

    # ── Section 1 ────────────────────────────────────────────────────────────
    lines += [SEP, "  SECTION 1 — CLINICAL TRIALS", SEP, ""]

    # 1.1
    lines += [
        SEP2, "  1.1  OVERVIEW & TRIAL TREND", SEP2, "",
        f"  [CHART]  {chart_paths.get('trend', 'chart_trend.png')}",
        "           Description: Line + area chart showing number of clinical",
        "           trials started per calendar year, with polynomial trend line.", "",
        "  SUMMARY STATISTICS",
        f"    Total Studies    : {data['total']}",
        f"    Completed        : {data['completed']}",
        f"    Ongoing / Active : {data['ongoing']}",
        f"    Withdrawn        : {data['withdrawn_count']}",
        f"    Other            : {data['other']}", "",
        "  STATUS BREAKDOWN",
    ]
    for status, cnt in sorted(data["status_counter"].items(), key=lambda x: -x[1]):
        lines.append(f"    {status:<35} {cnt}")
    lines.append("")

    # 1.2
    lines += [SEP2, "  1.2  ASSOCIATED CONDITIONS / DISEASES  (top 15)", SEP2, ""]
    top_conds = data["conditions"].most_common(15)
    if top_conds:
        lines.append(_table(["Condition", "Studies"], [], [52, 8]))
        lines.append("  ".join(["-" * 52, "-" * 8]))
        for cond, cnt in top_conds:
            lines.append(_col([cond, cnt], [52, 8]))
    else:
        lines.append("  No condition data available.")
    lines.append("")

    # 1.3
    lines += [SEP2, "  1.3  ALL INTERVENTIONS (unique entries from trial data)", SEP2, ""]
    interventions = data.get("interventions_all", [])
    if interventions:
        lines.append(_table(["Intervention Name", "NCT ID(s)"], [], [45, 40]))
        lines.append("  ".join(["-" * 45, "-" * 40]))
        for name, nct_ids_str in interventions:
            lines.append(_col([name, nct_ids_str], [45, 40]))
    else:
        lines.append("  No intervention data found.")
    lines.append("")

    # 1.4
    lines += [SEP2, "  1.4  COMBINATION THERAPIES / PRODUCTS", SEP2, ""]
    combo_rows = data.get("combo_rows", [])
    if combo_rows:
        lines.append(_table(["Interventions", "NCT ID"], [], [60, 15]))
        lines.append("  ".join(["-" * 60, "-" * 15]))
        for interventions_str, nct_id in combo_rows:
            lines.append(_col([interventions_str, nct_id], [60, 15]))
    else:
        lines.append("  No combination therapies identified.")
    lines.append("")

    # 1.5
    lines += [
        SEP2, "  1.5  LEAD SPONSORS — ACTIVITY & KEY STUDIES", SEP2, "",
        f"  [CHART]  {chart_paths.get('sponsors', 'chart_sponsors.png')}",
        "           Description: Mini panel charts (bar = annual count,",
        "           line = cumulative) for top 6 lead sponsors, last 10 yrs.", "",
        "  ALL SPONSORS (ranked by study count):",
        _table(["Sponsor", "Studies"], [], [52, 8]),
        "  ".join(["-" * 52, "-" * 8]),
    ]
    for sp, cnt in data["sponsors_full"].most_common():
        lines.append(_col([sp, cnt], [52, 8]))
    lines.append("")
    lines.append("  KEY STUDIES BY LEAD SPONSOR (last 10 years, up to 8 per sponsor)")
    lines.append("")
    for sp in data["top_sponsors"]:
        lines.append(f"  >>> {sp}")
        sl = sorted(data["sponsor_studies"].get(sp, []),
                    key=lambda x: parse_date(x.get("start", "")) or datetime.min,
                    reverse=True)[:8]
        if sl:
            lines.append("  " + _table(
                ["Study Title", "NCT ID", "Phase", "Status", "Start", "End"],
                [], [50, 14, 12, 22, 10, 10]
            ))
            lines.append("  " + "  ".join(["-"*50,"-"*14,"-"*12,"-"*22,"-"*10,"-"*10]))
            for sd in sl:
                t = sd["title"][:50] + ("…" if len(sd["title"]) > 50 else "")
                lines.append("  " + _col(
                    [t, sd["nct_id"], sd["phase"], sd["status"],
                     sd.get("start", "—") or "—", sd.get("end", "—") or "—"],
                    [50, 14, 12, 22, 10, 10]
                ))
        else:
            lines.append("    No studies in this window.")
        lines.append("")

    # 1.6
    lines += [
        SEP2, "  1.6  COLLABORATING INSTITUTIONS — ACTIVITY & KEY STUDIES", SEP2, "",
        f"  [CHART]  {chart_paths.get('collabs', 'chart_collabs.png')}",
        "           Description: Mini panel charts (bar = annual count,",
        "           line = cumulative) for top 6 collaborators, last 10 yrs.", "",
        "  ALL COLLABORATORS (ranked by study count):",
        _table(["Collaborator", "Studies"], [], [52, 8]),
        "  ".join(["-" * 52, "-" * 8]),
    ]
    for co, cnt in data["collaborators_full"].most_common():
        lines.append(_col([co, cnt], [52, 8]))
    lines.append("")
    lines.append("  KEY STUDIES BY COLLABORATOR (last 10 years, up to 8 per institution)")
    lines.append("")
    for co in data["top_collabs"]:
        lines.append(f"  >>> {co}")
        sl = sorted(data["collab_studies"].get(co, []),
                    key=lambda x: parse_date(x.get("start", "")) or datetime.min,
                    reverse=True)[:8]
        if sl:
            lines.append("  " + _table(
                ["Study Title", "NCT ID", "Phase", "Status", "Start", "End"],
                [], [50, 14, 12, 22, 10, 10]
            ))
            lines.append("  " + "  ".join(["-"*50,"-"*14,"-"*12,"-"*22,"-"*10,"-"*10]))
            for sd in sl:
                t = sd["title"][:50] + ("…" if len(sd["title"]) > 50 else "")
                lines.append("  " + _col(
                    [t, sd["nct_id"], sd["phase"], sd["status"],
                     sd.get("start", "—") or "—", sd.get("end", "—") or "—"],
                    [50, 14, 12, 22, 10, 10]
                ))
        else:
            lines.append("    No studies in this window.")
        lines.append("")

    # 1.7
    lines += [SEP2, "  1.7  WITHDRAWN STUDIES & REASONS", SEP2, ""]
    withdrawn = data["withdrawn_info"]
    if not withdrawn:
        lines.append("  No withdrawn studies found.")
    else:
        lines.append(_table(
            ["Study Title", "NCT ID", "Start", "Reason for Withdrawal"],
            [], [50, 14, 10, 50]
        ))
        lines.append("  ".join(["-"*50, "-"*14, "-"*10, "-"*50]))
        for w in withdrawn:
            t = w["title"][:50] + ("…" if len(w["title"]) > 50 else "")
            lines.append(_col(
                [t, w["nct_id"], w["start"] or "—", w["why"][:50]],
                [50, 14, 10, 50]
            ))
    lines.append("")

    # ── Footer ───────────────────────────────────────────────────────────────
    lines += [
        SEP,
        f"  END OF REPORT  —  {pn.upper()}",
        f"  Generated: {now}  |  Source: ClinicalTrials.gov",
        SEP, "",
        "CHART FILES REFERENCED IN THIS REPORT", SEP2,
    ]
    for key, path in chart_paths.items():
        lines.append(f"  {key:<15} → {path}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PATENTS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def get_epo_token():
    auth = base64.b64encode(f"{EPO_KEY}:{EPO_SECRET}".encode()).decode()
    r = requests.post(
        "https://ops.epo.org/3.2/auth/accesstoken",
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials"}
    )
    if r.status_code != 200:
        raise RuntimeError(f"EPO auth failed: {r.text}")
    return r.json()["access_token"]


def search_patents(token, protein, start, end):
    query = (
        f'ti="{protein}" and ab=inhibitor or '
        f'ti="{protein}" and ab=therapy or '
        f'ti="{protein}" and ab=antagonist'
    )
    r = requests.get(
        "https://ops.epo.org/3.2/rest-services/published-data/search",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={"q": query, "Range": f"{start}-{end}"}
    )
    if r.status_code == 404:
        return []
    if r.status_code != 200:
        print(f"  Search error ({r.status_code}): {r.text[:200]}")
        return []
    refs = (
        r.json().get("ops:world-patent-data", {})
                .get("ops:biblio-search", {})
                .get("ops:search-result", {})
                .get("ops:publication-reference", [])
    )
    return [refs] if isinstance(refs, dict) else refs


def get_biblio(token, ep_id):
    r = requests.get(
        f"https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{ep_id}/biblio",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
    )
    return r.json() if r.status_code == 200 else None


def extract_text(field):
    if isinstance(field, dict):
        return field.get("$", "")
    if isinstance(field, list) and field:
        return field[0].get("$", "") if isinstance(field[0], dict) else ""
    return ""


def parse_biblio(bdata, ep_id):
    exchange_docs = (
        bdata.get("ops:world-patent-data", {})
             .get("exchange-documents", {})
             .get("exchange-document", {})
    )
    if isinstance(exchange_docs, dict):
        exchange_docs = [exchange_docs]

    rows = []
    for doc in exchange_docs:
        bd = doc.get("bibliographic-data", {})

        titles = bd.get("invention-title", [])
        if isinstance(titles, dict): titles = [titles]
        title = next(
            (t.get("$", "") for t in titles if t.get("@lang") == "en"),
            titles[0].get("$", "") if titles else ""
        )

        pub_refs = bd.get("publication-reference", {})
        if isinstance(pub_refs, dict): pub_refs = [pub_refs]
        year = ""
        for pr in pub_refs:
            doc_ids = pr.get("document-id", [])
            if isinstance(doc_ids, dict): doc_ids = [doc_ids]
            for did in doc_ids:
                date = did.get("date", {}).get("$", "")
                if date:
                    year = date[:4]
                    break
            if year: break

        ipc_list = bd.get("patent-classifications", {}).get("patent-classification", [])
        if isinstance(ipc_list, dict): ipc_list = [ipc_list]
        ipc_codes = list({
            "".join([
                extract_text(i.get("section", {})),
                extract_text(i.get("class", {})),
                extract_text(i.get("subclass", {})),
                extract_text(i.get("main-group", {})),
                "/", extract_text(i.get("subgroup", {}))
            ]).strip("/")
            for i in ipc_list
        })

        applicants_raw = bd.get("parties", {}).get("applicants", {}).get("applicant", [])
        if isinstance(applicants_raw, dict): applicants_raw = [applicants_raw]
        sponsors_pat = list({
            a.get("applicant-name", {}).get("name", {}).get("$", "")
            for a in applicants_raw
            if a.get("applicant-name", {}).get("name", {}).get("$", "")
        })

        inventors_raw = bd.get("parties", {}).get("inventors", {}).get("inventor", [])
        if isinstance(inventors_raw, dict): inventors_raw = [inventors_raw]
        inventors = list({
            i.get("inventor-name", {}).get("name", {}).get("$", "")
            for i in inventors_raw
            if i.get("inventor-name", {}).get("name", {}).get("$", "")
        })

        abstracts = bd.get("abstract", [])
        if isinstance(abstracts, dict): abstracts = [abstracts]
        abstract_text = ""
        for a in abstracts:
            if a.get("@lang") == "en":
                p = a.get("p", "")
                abstract_text = p.get("$", "") if isinstance(p, dict) else " ".join(
                    x.get("$", "") for x in p if isinstance(x, dict)
                )
                break

        mol_hits = re.findall(
            r'\b([A-Z]{2,8}[\d\-]*[A-Z\d]*'
            r'|[a-z]+(inib|afib|mab|zumab|tinib|ciclib|lisib|parib|enib|umab|ximab|rafenib|cetinib))\b',
            abstract_text
        )
        molecules = list({m[0] for m in mol_hits}) if mol_hits else []

        rows.append({
            "Patent ID": ep_id,
            "Title":     title,
            "Year":      year,
            "Molecules": "; ".join(molecules),
            "Sponsors":  "; ".join(sponsors_pat),
            "Inventors": "; ".join(inventors),
            "IPC Codes": "; ".join(ipc_codes),
            "Abstract":  abstract_text[:500] + ("..." if len(abstract_text) > 500 else "")
        })
    return rows


def run_patents(protein, output_csv):
    print(f"\n[PATENTS] Fetching patents for: {protein}")
    token = get_epo_token()
    print("  EPO authenticated")

    all_refs = []
    for start in range(1, EPO_MAX_RESULTS + 1, EPO_BATCH_SIZE):
        end = min(start + EPO_BATCH_SIZE - 1, EPO_MAX_RESULTS)
        print(f"  Fetching {start}–{end}...", end=" ")
        batch = search_patents(token, protein, start, end)
        if not batch:
            print("no more results.")
            break
        all_refs.extend(batch)
        print(f"{len(batch)} refs")
        time.sleep(0.5)

    print(f"  Total references: {len(all_refs)}")
    all_rows = []

    for idx, ref in enumerate(all_refs, 1):
        doc_id  = ref.get("document-id", {})
        if isinstance(doc_id, list): doc_id = doc_id[0]
        country = doc_id.get("country", {}).get("$", "")
        doc_num = doc_id.get("doc-number", {}).get("$", "")
        kind    = doc_id.get("kind", {}).get("$", "")
        ep_id   = f"{country}{doc_num}.{kind}"

        print(f"  [{idx}/{len(all_refs)}] {ep_id}", end=" ")
        bdata = get_biblio(token, ep_id)
        if not bdata:
            print("skipped")
            continue
        rows = parse_biblio(bdata, ep_id)
        all_rows.extend(rows)
        print("ok")
        time.sleep(EPO_DELAY_SECONDS)

    if not all_rows:
        print("  No patent data extracted.")
        return []

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Patent ID", "Title", "Year", "Molecules",
            "Sponsors", "Inventors", "IPC Codes", "Abstract"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Patents CSV saved → {output_csv}  ({len(all_rows)} rows)")
    return all_rows


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_report(protein_name, output_dir=None, df_selective=None):
    if output_dir is None:
        output_dir = re.sub(r"[^\w]", "_", protein_name)

    images_dir = os.path.join(output_dir, "images")
    txts_dir   = os.path.join(output_dir, "txts")
    csvs_dir   = os.path.join(output_dir, "csvs")
    for d in (images_dir, txts_dir, csvs_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}\n  Target Landscape Report\n  Protein: {protein_name}\n{'='*60}")

    print("  Fetching studies from ClinicalTrials.gov…")
    studies = fetch_and_merge_studies(protein_name, df_selective)
    if not studies:
        print("  No studies found. Exiting.")
        return None

    print("  Processing data…")
    data = process_studies(studies, protein_name)

    print("  Saving charts…")
    years_10 = list(range(data["cutoff_10yr"], data["now_year"] + 1))
    chart_paths = {
        "trend": save_trend_chart(data, images_dir),
        "sponsors": save_entity_panel_chart(
            data["sponsor_year_count"], data["top_sponsors"], years_10,
            "Lead Sponsors — Studies per Year (last 10 yrs)",
            "chart_sponsors.png", images_dir,
        ),
        "collabs": save_entity_panel_chart(
            data["collab_year_count"], data["top_collabs"], years_10,
            "Collaborating Institutions — Studies per Year (last 10 yrs)",
            "chart_collabs.png", images_dir,
        ),
    }

    print("  Writing text report…")
    safe_name = re.sub(r"[^\w]", "_", protein_name)
    txt_path  = os.path.join(txts_dir, f"{safe_name}_landscape_report.txt")
    write_txt_report(data, chart_paths, txt_path)

    print("  Writing trials CSV…")
    trials_csv = os.path.join(csvs_dir, f"{safe_name}_trials.csv")
    write_trials_csv(data, trials_csv)

    patents_csv = os.path.join(csvs_dir, f"{safe_name}_patents.csv")
    run_patents(protein_name, patents_csv)

    all_files = [txt_path, trials_csv, patents_csv] + list(chart_paths.values())
    print(f"\n  Output folder : {output_dir}/")
    print("  Files created :")
    for f in all_files:
        print(f"    {f}")

    return {
        "txt_path":   txt_path,
        "trials_csv": trials_csv,
        "patents_csv": patents_csv,
        "_ct_data": data,
    }