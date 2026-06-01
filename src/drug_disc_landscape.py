
import re
import time
import json
import requests
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
EUTILS_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OLLAMA_URL = "http://localhost:11434/api/generate"

QWEN_MODEL = "qwen2.5:7b-instruct"
QWEN_MODEL     = "qwen2.5:7b-instruct"
MAX_PER_QUERY  = 50    # PubMed esearch retmax
TOP_N          = 10    # articles passed to filter/synthesis per dimension
RECENCY_CUTOFF = 2022  # papers >= this year get 2x weight in signal score
ACTIVE_CUTOFF  = 2020  # >= 2 papers post this year → "Active"
EMERGING_CUTOFF= 2020  # 1 paper post this year     → "Emerging"


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSION REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# Each dimension entry:
#   group          — high-level area (used for report sections)
#   label          — short display name
#   why_it_matters — one sentence: why this dimension shapes study design
#   query_terms    — PubMed query terms ([ti] or [TIAB])
#   anchor_words   — checked in TITLE for Stage-1 pre-filter
#   synthesis_context — sent to LLM to frame the synthesis task
# ══════════════════════════════════════════════════════════════════════════════

DIMENSIONS = {

    # ── BINDING SITE ──────────────────────────────────────────────────────────
    "site_orthosteric": {
        "group": "Binding Site",
        "label": "Orthosteric / Active Site",
        "why_it_matters": (
            "Defines whether the primary catalytic or ligand-binding pocket is "
            "druggable and what chemical features it requires."
        ),
        "query_terms": [
            '"active site"[ti]', '"orthosteric"[ti]', '"ATP-binding site"[ti]',
            '"substrate binding"[ti]', '"catalytic site"[ti]',
            '"active-site inhibitor"[ti]',
        ],
        "anchor_words": [
            "orthosteric", "active site", "active-site", "atp-binding",
            "catalytic site", "substrate binding",
        ],
        "synthesis_context": (
            "These papers characterise the primary/orthosteric binding site of the target. "
            "Summarise: what is established about its druggability, and what does this imply "
            "for a new inhibitor program?"
        ),
    },

    "site_allosteric": {
        "group": "Binding Site",
        "label": "Allosteric Site / Modulator",
        "why_it_matters": (
            "Allosteric sites can confer selectivity advantages and bypass "
            "resistance mechanisms that affect the orthosteric site."
        ),
        "query_terms": [
            '"allosteric"[ti]', '"allosteric modulator"[TIAB]',
            '"allosteric inhibitor"[TIAB]', '"positive allosteric"[TIAB]',
            '"negative allosteric"[TIAB]', '"PAM"[TIAB]', '"NAM"[TIAB]',
            '"allosteric site"[TIAB]',
        ],
        "anchor_words": [
            "allosteric", "pam", "nam", "positive allosteric", "negative allosteric",
        ],
        "synthesis_context": (
            "These papers investigate allosteric sites or modulators of the target. "
            "Summarise: which allosteric mechanisms have been validated, and does "
            "the literature suggest allosteric targeting is a preferred strategy?"
        ),
    },

    "site_cryptic": {
        "group": "Binding Site",
        "label": "Cryptic / Alternative Pocket",
        "why_it_matters": (
            "Cryptic pockets that open on ligand binding can be highly selective "
            "handles unavailable to competing programs."
        ),
        "query_terms": [
            '"cryptic site"[TIAB]', '"cryptic pocket"[TIAB]',
            '"alternative binding site"[TIAB]', '"secondary binding site"[TIAB]',
            '"hidden pocket"[TIAB]', '"induced-fit pocket"[TIAB]',
            '"druggable pocket"[ti]', '"binding groove"[ti]',
        ],
        "anchor_words": [
            "cryptic", "hidden pocket", "alternative binding",
            "secondary binding", "binding groove", "druggable pocket",
        ],
        "synthesis_context": (
            "These papers describe non-orthosteric or cryptic pockets in the target. "
            "Summarise: what alternative pockets have been found, and what opportunity "
            "do they represent for differentiated programs?"
        ),
    },

    # ── BINDING MODE ──────────────────────────────────────────────────────────
    "mode_covalent": {
        "group": "Binding Mode",
        "label": "Covalent / Irreversible",
        "why_it_matters": (
            "If a reactive cysteine or lysine is accessible, covalent inhibitors "
            "can achieve long residence time and overcome certain resistance mutations."
        ),
        "query_terms": [
            '"covalent inhibitor"[TIAB]', '"irreversible inhibitor"[TIAB]',
            '"covalent binding"[TIAB]', '"warhead"[TIAB]',
            '"electrophilic inhibitor"[TIAB]', '"covalent"[ti]',
            '"cysteine-targeted"[TIAB]',
        ],
        "anchor_words": [
            "covalent", "irreversible inhibitor", "warhead",
            "electrophilic", "cysteine-targeted",
        ],
        "synthesis_context": (
            "These papers describe covalent or irreversible inhibitors of the target. "
            "Summarise: which reactive residues are targeted, what warhead chemistries "
            "are used, and is covalent inhibition considered a viable strategy?"
        ),
    },

    "mode_type2": {
        "group": "Binding Mode",
        "label": "Type II / DFG-out / Inactive State",
        "why_it_matters": (
            "Type II binding exploits the DFG-out inactive conformation, often "
            "enabling selectivity by accessing a hydrophobic back pocket absent "
            "in active-state structures."
        ),
        "query_terms": [
            '"type II inhibitor"[ti]', '"DFG-out"[TIAB]',
            '"inactive conformation"[ti]', '"type-II"[ti]',
            '"C-helix out"[TIAB]', '"back pocket"[TIAB]',
        ],
        "anchor_words": [
            "type ii", "type-ii", "dfg-out", "inactive conformation",
            "c-helix out", "back pocket",
        ],
        "synthesis_context": (
            "These papers describe Type II or inactive-state binding to the target. "
            "Summarise: is this binding mode validated, and what selectivity or "
            "resistance advantages does it offer?"
        ),
    },

    # ── MODALITY ──────────────────────────────────────────────────────────────
    "modality_degrader": {
        "group": "Modality",
        "label": "Degraders / PROTACs / Molecular Glues",
        "why_it_matters": (
            "Degraders eliminate the protein scaffold — important when the target's "
            "non-enzymatic functions drive disease or when inhibitor resistance is common."
        ),
        "query_terms": [
            '"PROTAC"[TIAB]', '"targeted protein degradation"[TIAB]',
            '"molecular glue"[TIAB]', '"bifunctional degrader"[TIAB]',
            '"degrader"[ti]', '"E3 ligase"[TIAB]', '"cereblon"[TIAB]',
            '"VHL recruiter"[TIAB]',
        ],
        "anchor_words": [
            "protac", "degrader", "molecular glue",
            "targeted protein degradation", "cereblon", "vhl recruiter",
        ],
        "synthesis_context": (
            "These papers describe degrader or PROTAC approaches to the target. "
            "Summarise: is degradation validated over inhibition, which E3 ligases "
            "are used, and what disease rationale drives this modality?"
        ),
    },

    "modality_macrocycle": {
        "group": "Modality",
        "label": "Macrocycle / Cyclic Peptide",
        "why_it_matters": (
            "Macrocycles access flat or featureless binding surfaces (e.g. PPIs) "
            "that are intractable to standard small molecules."
        ),
        "query_terms": [
            '"macrocycle"[TIAB]', '"macrocyclic"[TIAB]', '"cyclic peptide"[TIAB]',
            '"stapled peptide"[TIAB]', '"constrained peptide"[TIAB]',
            '"bicyclic peptide"[TIAB]',
        ],
        "anchor_words": [
            "macrocycle", "macrocyclic", "cyclic peptide",
            "stapled peptide", "constrained peptide", "bicyclic peptide",
        ],
        "synthesis_context": (
            "These papers describe macrocyclic or cyclic peptide approaches to the target. "
            "Summarise: why is this modality preferred here, and what binding surface "
            "or challenge justifies it over standard small molecules?"
        ),
    },

    "modality_ppi": {
        "group": "Modality",
        "label": "PPI Inhibitor / Disruptor",
        "why_it_matters": (
            "If the target's disease role is primarily mediated through a "
            "protein-protein interface rather than enzymatic activity, PPI "
            "disruption may be more relevant than catalytic inhibition."
        ),
        "query_terms": [
            '"protein-protein interaction"[ti]', '"PPI inhibitor"[TIAB]',
            '"PPI disruptor"[TIAB]', '"interface inhibitor"[TIAB]',
            '"hot spot"[ti]', '"protein interaction inhibitor"[ti]',
        ],
        "anchor_words": [
            "protein-protein interaction", "ppi inhibitor", "ppi disruptor",
            "interface inhibitor", "hot spot",
        ],
        "synthesis_context": (
            "These papers describe PPI inhibition at the target. "
            "Summarise: which protein-protein interface is targeted, what "
            "chemical approaches are taken, and is this the primary strategy?"
        ),
    },

    # ── SELECTIVITY STRATEGY ──────────────────────────────────────────────────
    "selectivity_isoform": {
        "group": "Selectivity Strategy",
        "label": "Isoform / Paralog Selectivity",
        "why_it_matters": (
            "If isoform selectivity is therapeutically required, it constrains "
            "the binding site and chemical space from the outset."
        ),
        "query_terms": [
            '"isoform selectivity"[TIAB]', '"isoform-selective"[ti]',
            '"selectivity"[ti]', '"paralog selectivity"[TIAB]',
            '"subtype-selective"[ti]',
        ],
        "anchor_words": [
            "isoform selectiv", "isoform-selective",
            "subtype-selective", "paralog selectiv",
        ],
        "synthesis_context": (
            "These papers address isoform or paralog selectivity for the target. "
            "Summarise: which paralogs must be avoided, what selectivity is achievable, "
            "and what structural features enable discrimination?"
        ),
    },

    "selectivity_dual": {
        "group": "Selectivity Strategy",
        "label": "Dual / Multi-Target",
        "why_it_matters": (
            "Published dual-target rationales tell you which co-target combinations "
            "have synergistic biological evidence, saving you from building "
            "a polypharmacology hypothesis from scratch."
        ),
        "query_terms": [
            '"dual inhibitor"[ti]', '"dual targeting"[ti]',
            '"multi-target"[ti]', '"polypharmacology"[ti]',
            '"dual-acting"[ti]', '"bifunctional inhibitor"[ti]',
            '"co-inhibition"[TIAB]', '"simultaneous inhibition"[ti]',
        ],
        "anchor_words": [
            "dual inhibitor", "dual targeting", "dual-acting",
            "multi-target", "polypharmacology", "bifunctional inhibitor",
            "co-inhibition", "simultaneous inhibition",
        ],
        "synthesis_context": (
            "These papers describe dual or multi-target approaches involving the target. "
            "Summarise: which co-targets are paired, what biological rationale drives "
            "the combination, and what does this imply for single-agent design?"
        ),
    },

    "selectivity_panel": {
        "group": "Selectivity Strategy",
        "label": "Selectivity Profiling / Off-Target Panel",
        "why_it_matters": (
            "Published off-target profiles tell you which family members or unrelated "
            "targets have been hit by existing compounds — these define the minimum "
            "selectivity panel for your program."
        ),
        "query_terms": [
            '"selectivity profile"[TIAB]', '"selectivity profiling"[TIAB]',
            '"off-target"[ti]', '"counter-screen"[TIAB]',
            '"kinome selectivity"[TIAB]', '"panel selectivity"[TIAB]',
        ],
        "anchor_words": [
            "selectivity profile", "selectivity profiling", "off-target",
            "counter-screen", "kinome selectivity", "panel selectivity",
        ],
        "synthesis_context": (
            "These papers profile selectivity or off-targets for the target. "
            "Summarise: which off-targets are most commonly hit, what panel is "
            "considered standard, and what are the key selectivity challenges?"
        ),
    },

    # ── MUTATIONS & VARIANTS ──────────────────────────────────────────────────
    "mutation_activating": {
        "group": "Mutations & Variants",
        "label": "Activating / Driver Mutations",
        "why_it_matters": (
            "Specific driver mutations may open unique pockets, change cofactor "
            "dependency, or require mutation-specific compounds entirely."
        ),
        "query_terms": [
            '"mutation"[ti]', '"mutant"[ti]', '"variant"[ti]',
            '"gain of function"[TIAB]', '"activating mutation"[TIAB]',
            '"oncogenic mutation"[TIAB]', '"driver mutation"[TIAB]',
        ],
        "anchor_words": [
            "mutation", "mutant", "variant", "gain of function",
            "activating mutation", "oncogenic mutation", "driver mutation",
        ],
        "synthesis_context": (
            "These papers focus on activating or driver mutations in the target. "
            "Summarise: which mutations are the primary focus, do they create "
            "unique binding opportunities, and should a program be allele-specific?"
        ),
    },

    "mutation_resistance": {
        "group": "Mutations & Variants",
        "label": "Resistance Mutations",
        "why_it_matters": (
            "Knowing which resistance mutations emerge early lets you design "
            "compounds that retain activity against the most likely escape alleles."
        ),
        "query_terms": [
            '"drug resistance"[ti]', '"acquired resistance"[ti]',
            '"gatekeeper mutation"[TIAB]', '"resistance mutation"[TIAB]',
            '"resistance mechanism"[ti]', '"overcome resistance"[ti]',
            '"solvent-front mutation"[TIAB]',
        ],
        "anchor_words": [
            "drug resistance", "acquired resistance", "gatekeeper",
            "resistance mutation", "overcome resistance", "solvent-front",
        ],
        "synthesis_context": (
            "These papers describe resistance mutations or mechanisms for the target. "
            "Summarise: which mutations are most clinically significant, what "
            "structural changes do they cause, and what design strategies address them?"
        ),
    },

    # ── COMBINATION THERAPY ───────────────────────────────────────────────────
    "combo_synergy": {
        "group": "Combination Therapy",
        "label": "Combination / Synergy",
        "why_it_matters": (
            "If the field already has validated combination rationales, your "
            "compound's selectivity profile and PK must be compatible with the "
            "intended partner agent."
        ),
        "query_terms": [
            '"combination"[ti]', '"synergy"[ti]', '"synergistic"[ti]',
            '"co-treatment"[TIAB]', '"combined inhibition"[ti]',
            '"drug combination"[ti]', '"combination therapy"[ti]',
        ],
        "anchor_words": [
            "combination", "synerg", "co-treatment",
            "combined inhibition", "drug combination",
        ],
        "synthesis_context": (
            "These papers describe combination strategies involving the target. "
            "Summarise: which partner agents are most commonly combined, what is "
            "the biological rationale, and what does this mean for compound design?"
        ),
    },

    "combo_synthetic_lethality": {
        "group": "Combination Therapy",
        "label": "Synthetic Lethality / Pathway Co-targeting",
        "why_it_matters": (
            "Synthetic lethal relationships define which pathway partners must "
            "be considered when designing a combination program."
        ),
        "query_terms": [
            '"synthetic lethality"[TIAB]', '"synthetic lethal"[TIAB]',
            '"co-targeting"[ti]', '"feedback bypass"[TIAB]',
            '"vertical inhibition"[TIAB]', '"pathway combination"[ti]',
        ],
        "anchor_words": [
            "synthetic lethality", "synthetic lethal",
            "co-targeting", "feedback bypass", "pathway combination",
        ],
        "synthesis_context": (
            "These papers describe synthetic lethality or co-targeting strategies "
            "for the target. Summarise: which synthetic lethal partners are validated, "
            "and what does this imply for combination program design?"
        ),
    },

    # ── STRUCTURAL BIOLOGY ────────────────────────────────────────────────────
    "structural_structure": {
        "group": "Structural Biology",
        "label": "Crystal / Cryo-EM Structure",
        "why_it_matters": (
            "The quality and availability of structural data determines whether "
            "structure-based design is feasible and which conformational states "
            "can be targeted."
        ),
        "query_terms": [
            '"crystal structure"[TIAB]', '"X-ray structure"[TIAB]',
            '"cryo-EM"[TIAB]', '"co-crystal structure"[TIAB]',
            '"NMR structure"[TIAB]', '"structure determination"[ti]',
        ],
        "anchor_words": [
            "crystal structure", "x-ray structure", "cryo-em",
            "co-crystal", "nmr structure", "structure determination",
        ],
        "synthesis_context": (
            "These papers report structures of the target. "
            "Summarise: how well is the structural landscape covered (apo, holo, "
            "different conformations), and what gaps remain for structure-based design?"
        ),
    },

    "structural_binding_mode": {
        "group": "Structural Biology",
        "label": "Binding Mode / Interaction Analysis",
        "why_it_matters": (
            "Binding mode data tells you which pharmacophoric features are "
            "non-negotiable and which vectors are available for optimisation."
        ),
        "query_terms": [
            '"binding mode"[ti]', '"binding pose"[TIAB]',
            '"binding conformation"[TIAB]', '"interaction mode"[ti]',
            '"key interactions"[ti]', '"binding geometry"[TIAB]',
        ],
        "anchor_words": [
            "binding mode", "binding pose", "binding conformation",
            "interaction mode", "key interactions", "binding geometry",
        ],
        "synthesis_context": (
            "These papers characterise how compounds bind to the target. "
            "Summarise: what binding modes are described, which interactions are "
            "conserved across chemotypes, and what design implications follow?"
        ),
    },

    # ── HIT DISCOVERY & OPTIMISATION ─────────────────────────────────────────
    "discovery_fragment_hts": {
        "group": "Hit Discovery & Optimisation",
        "label": "Fragment-Based / HTS",
        "why_it_matters": (
            "Published fragment and HTS data defines the accessible chemical "
            "starting space and can reveal binding modes not seen with larger compounds."
        ),
        "query_terms": [
            '"fragment-based"[TIAB]', '"FBDD"[TIAB]', '"fragment screening"[TIAB]',
            '"high-throughput screening"[TIAB]', '"HTS"[ti]',
            '"hit-to-lead"[TIAB]', '"hit compound"[ti]',
        ],
        "anchor_words": [
            "fragment-based", "fbdd", "fragment screening",
            "high-throughput screening", "hit-to-lead", "hit compound",
        ],
        "synthesis_context": (
            "These papers describe fragment or HTS campaigns for the target. "
            "Summarise: what hit rates and chemotypes emerged, and what does "
            "the hit landscape imply for tractability?"
        ),
    },

    "discovery_sar": {
        "group": "Hit Discovery & Optimisation",
        "label": "SAR / Lead Optimisation",
        "why_it_matters": (
            "The published SAR tells you which chemical vectors tolerate "
            "modification and which are critical for potency — avoiding "
            "rediscovering constraints already mapped."
        ),
        "query_terms": [
            '"structure-activity relationship"[ti]', '"SAR"[ti]',
            '"QSAR"[ti]', '"lead optimisation"[ti]',
            '"lead optimization"[ti]', '"medicinal chemistry"[ti]',
            '"bioisostere"[TIAB]', '"prodrug"[ti]',
        ],
        "anchor_words": [
            "structure-activity", "sar study", "qsar", "lead optimi",
            "medicinal chemistry", "bioisostere", "prodrug",
        ],
        "synthesis_context": (
            "These papers describe SAR campaigns for the target. "
            "Summarise: which chemical series have been most developed, "
            "what are the key SAR lessons, and what optimisation challenges recur?"
        ),
    },

    # ── PRECLINICAL & CLINICAL ────────────────────────────────────────────────
    "progression_clinical": {
        "group": "Preclinical & Clinical",
        "label": "Clinical Candidate / Trial",
        "why_it_matters": (
            "Knowing which compounds have reached the clinic tells you what "
            "property benchmarks (potency, selectivity, PK) are achievable "
            "and what the competitive landscape looks like."
        ),
        "query_terms": [
            '"clinical candidate"[ti]', '"Phase I"[ti]', '"Phase II"[ti]',
            '"Phase III"[ti]', '"clinical trial"[ti]',
            '"first-in-class"[ti]', '"preclinical candidate"[ti]',
        ],
        "anchor_words": [
            "clinical candidate", "phase i", "phase ii", "phase iii",
            "clinical trial", "first-in-class", "preclinical candidate",
        ],
        "synthesis_context": (
            "These papers describe clinical or preclinical candidates for the target. "
            "Summarise: which compounds have progressed furthest, what property "
            "profiles enabled progression, and what is the competitive landscape?"
        ),
    },

    # ── FUNCTIONAL PHARMACOLOGY ───────────────────────────────────────────────
    "functional_bias": {
        "group": "Functional Pharmacology",
        "label": "Biased Agonism / Functional Selectivity",
        "why_it_matters": (
            "For GPCRs and other multi-effector targets, biased signalling data "
            "tells you whether G protein vs arrestin pathway selectivity is a "
            "therapeutic objective or a liability."
        ),
        "query_terms": [
            '"functional selectivity"[TIAB]', '"biased agonism"[TIAB]',
            '"biased signaling"[TIAB]', '"biased signalling"[TIAB]',
            '"beta-arrestin bias"[TIAB]', '"G protein bias"[TIAB]',
            '"pathway-selective"[TIAB]',
        ],
        "anchor_words": [
            "biased agonism", "biased signaling", "biased signalling",
            "functional selectivity", "beta-arrestin bias",
            "g protein bias", "pathway-selective",
        ],
        "synthesis_context": (
            "These papers describe biased agonism or functional selectivity at the target. "
            "Summarise: which signalling pathway is therapeutically preferred, "
            "and how should this guide compound profiling?"
        ),
    },
}

# ── Activity status thresholds ─────────────────────────────────────────────────
def activity_status(articles: list[dict]) -> str:
    """
    Active    — ≥ 2 papers published >= ACTIVE_CUTOFF
    Emerging  — 1 paper published >= EMERGING_CUTOFF
    Historical— papers exist but all before cutoff
    Unexplored— no papers at all
    """
    if not articles:
        return "Unexplored"
    recent = [a for a in articles if _year_int(a) >= ACTIVE_CUTOFF]
    if len(recent) >= 2:
        return "Active"
    if len(recent) == 1:
        return "Emerging"
    return "Historical"


def signal_score(articles: list[dict]) -> float:
    """
    Volume × recency weight.
    Papers >= RECENCY_CUTOFF count double.
    Normalised to number of articles so score is comparable across dimensions.
    """
    if not articles:
        return 0.0
    score = sum(2.0 if _year_int(a) >= RECENCY_CUTOFF else 1.0 for a in articles)
    return round(score, 1)


def _year_int(article: dict) -> int:
    try:
        return int(article.get("year") or 0)
    except (ValueError, TypeError):
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# QUERY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_queries(target: str, aliases: list[str] | None = None) -> dict[str, str]:
    aliases = aliases or []
    all_names = [target] + aliases
    name_parts = " OR ".join(f'"{n}"[TIAB]' for n in all_names)
    target_block = f"({name_parts})"

    queries = {}
    for dim_key, dim in DIMENSIONS.items():
        term_block = " OR ".join(dim["query_terms"])
        queries[dim_key] = (
            f'{target_block} AND ({term_block}) NOT "review"[PT]'
        )
    return queries


# ══════════════════════════════════════════════════════════════════════════════
# PUBMED FETCH
# ══════════════════════════════════════════════════════════════════════════════
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """Session with automatic retry + exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,          # waits 1, 2, 4, 8, 16 s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = make_session()

def esearch(query: str, api_key: str, max_results: int = MAX_PER_QUERY) -> list[str]:
    params = {
        "db": "pubmed", "term": query,
        "retmax": max_results, "retmode": "json", "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key
    for attempt in range(4):
        try:
            resp = SESSION.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=20)
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n    [esearch] attempt {attempt+1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)
    return []


def efetch_titles(pmids: list[str], api_key: str) -> list[dict]:
    if not pmids:
        return []
    params = {
        "db": "pubmed", "id": ",".join(pmids),
        "retmode": "xml", "rettype": "abstract",
    }
    if api_key:
        params["api_key"] = api_key
    for attempt in range(4):
        try:
            resp = SESSION.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=40)
            resp.raise_for_status()
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n    [efetch] attempt {attempt+1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)
    else:
        return []   # all retries exhausted
    # ... rest of XML parsing unchanged


def efetch_titles(pmids: list[str], api_key: str) -> list[dict]:
    if not pmids:
        return []
    params = {
        "db": "pubmed", "id": ",".join(pmids),
        "retmode": "xml", "rettype": "abstract",
    }
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    xml = resp.text

    articles = []
    for pmid, block in zip(
        re.findall(r"<PMID[^>]*>(\d+)</PMID>", xml),
        re.split(r"<PubmedArticle>", xml)[1:],
    ):
        title   = re.sub(r"<[^>]+>", "", _xml_text(block, "ArticleTitle")).strip()
        year    = _xml_text(block, "Year") or ""
        if not year:
            md   = _xml_text(block, "MedlineDate") or ""
            m    = re.search(r"\b(19|20)\d{2}\b", md)
            year = m.group() if m else ""
        journal = re.sub(r"<[^>]+>", "",
                         _xml_text(block, "Title") or
                         _xml_text(block, "ISOAbbreviation") or "").strip()
        if title:
            articles.append({"pmid": pmid, "title": title,
                              "year": year, "journal": journal})
    return articles


def _xml_text(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.DOTALL)
    return m.group(1).strip() if m else ""


def fetch_all(queries: dict[str, str], api_key: str) -> dict[str, list[dict]]:
    """Fetch all dimensions. Global PMID deduplication."""
    seen: set[str] = set()
    results: dict[str, list[dict]] = {}

    for label, query in queries.items():
        print(f"  [{label}] ...", end=" ", flush=True)
        pmids     = esearch(query, api_key)
        new_pmids = [p for p in pmids if p not in seen]
        seen.update(new_pmids)
        if new_pmids:
            arts = efetch_titles(new_pmids[:TOP_N], api_key)
            results[label] = arts
            print(f"{len(arts)} articles")
        else:
            results[label] = []
            print("0 (all seen in prior dimensions)")
        time.sleep(0.35)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — TITLE ANCHOR PRE-FILTER
# ══════════════════════════════════════════════════════════════════════════════

def title_has_anchor(title: str, anchor_words: list[str]) -> bool:
    t = title.lower()
    return any(a.lower() in t for a in anchor_words)


def prefilter_all(raw: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """
    Keep only articles whose title contains a dimension anchor word.
    No LLM involved. Fast and precise.
    """
    filtered = {}
    for dim_key, articles in raw.items():
        anchors = DIMENSIONS[dim_key]["anchor_words"]
        filtered[dim_key] = [
            a for a in articles
            if a.get("title") and title_has_anchor(a["title"], anchors)
        ]
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LLM SYNTHESIS (not classification)
# ══════════════════════════════════════════════════════════════════════════════

SYNTHESIS_PROMPT = """\
You are a medicinal chemistry expert helping a drug discovery team.

Target protein: {target}
Dimension: {label}
Why this matters: {why_it_matters}

The following paper titles represent all published work on this dimension \
for this target:

{title_list}

Task: {synthesis_context}

Write exactly 3 sentences:
1. What the field has established about this dimension for this target.
2. How active or mature this area is (based on the paper titles and dates provided).
3. One concrete implication for designing a NEW drug discovery program for this target.

Be specific to this target. Do not be generic. If the title list is very short, \
acknowledge limited evidence."""


def llm_synthesize(
    articles: list[dict],
    dim_key: str,
    target: str,
    model: str = QWEN_MODEL,
    url: str = OLLAMA_URL,
) -> str:
    dim    = DIMENSIONS[dim_key]
    titles = "\n".join(
        f"  [{a.get('year','?')}] {a['title']}"
        for a in sorted(articles, key=_year_int)
    )
    prompt = SYNTHESIS_PROMPT.format(
        target           = target,
        label            = dim["label"],
        why_it_matters   = dim["why_it_matters"],
        title_list       = titles,
        synthesis_context= dim["synthesis_context"],
    )
    try:
        resp = requests.post(url, json={
            "model":   model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.2, "num_predict": 200},
        }, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Synthesis unavailable: {e}]"


def synthesize_all(
    filtered: dict[str, list[dict]],
    target: str,
    use_llm: bool = True,
) -> dict[str, str]:
    """
    Returns {dim_key: synthesis_text} for dimensions with ≥1 article.
    """
    syntheses: dict[str, str] = {}
    populated = [(k, v) for k, v in filtered.items() if v]
    print(f"\n  Synthesising {len(populated)} populated dimensions ...")

    for dim_key, articles in populated:
        label = DIMENSIONS[dim_key]["label"]
        print(f"  [{label}] ...", end=" ", flush=True)
        if use_llm:
            text = llm_synthesize(articles, dim_key, target)
        else:
            text = (
                f"{len(articles)} paper(s) found "
                f"(earliest {min(_year_int(a) for a in articles)}, "
                f"latest {max(_year_int(a) for a in articles)}). "
                f"LLM synthesis disabled."
            )
        syntheses[dim_key] = text
        print("done")

    return syntheses


# ══════════════════════════════════════════════════════════════════════════════
# CO-OCCURRENCE — which PMIDs appear in multiple dimensions
# ══════════════════════════════════════════════════════════════════════════════

def find_cooccurrence(filtered: dict[str, list[dict]]) -> dict[str, list[str]]:
    """
    Returns {pmid: [dim_label, ...]} for PMIDs that appear in ≥2 dimensions.
    These cross-cutting papers are highest-signal for study design.
    Note: global dedup in fetch_all means true co-occurrence is rare but
    possible when a paper matches anchors in multiple dimensions.
    """
    from collections import defaultdict
    pmid_to_dims: dict[str, list[str]] = defaultdict(list)
    for dim_key, articles in filtered.items():
        label = DIMENSIONS[dim_key]["label"]
        for a in articles:
            pmid_to_dims[a["pmid"]].append(label)
    return {
        pmid: dims
        for pmid, dims in pmid_to_dims.items()
        if len(dims) >= 2
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORING & RANKING
# ══════════════════════════════════════════════════════════════════════════════

def rank_dimensions(filtered: dict[str, list[dict]]) -> list[tuple[str, float, str]]:
    """
    Returns list of (dim_key, score, status) sorted by score descending.
    """
    ranked = []
    for dim_key, articles in filtered.items():
        score  = signal_score(articles)
        status = activity_status(articles)
        ranked.append((dim_key, score, status))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT — decision-support report
# ══════════════════════════════════════════════════════════════════════════════

STATUS_ICONS = {
    "Active":      "●",   # solid — field is investing here now
    "Emerging":    "◑",   # half  — nascent activity
    "Historical":  "○",   # empty — studied but dormant
    "Unexplored":  "✗",   # cross — nobody has looked
}


def build_report(
    filtered:    dict[str, list[dict]],
    syntheses:   dict[str, str],
    target:      str,
    out_prefix:  str = "",
) -> None:
    safe    = re.sub(r"[^\w]", "_", target)
    ranked  = rank_dimensions(filtered)
    cooccur = find_cooccurrence(filtered)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Only keep Active / Emerging dimensions ──────────────────────────────
    INCLUDE_STATUSES = {"Active", "Emerging"}
    visible = [
        (dim_key, score, status)
        for dim_key, score, status in ranked
        if status in INCLUDE_STATUSES
    ]

    W = 74

    def rule(char="═"): return char * W
    def thin(): return "─" * W

    STATUS_LABEL = {
        "Active":   "● ACTIVE",
        "Emerging": "◑ EMERGING",
    }

    # ════════════════════════════════════════════════════════════════════════
    # TEXT REPORT
    # ════════════════════════════════════════════════════════════════════════
    txt_path = f"{safe}/txts/{out_prefix}{safe}_landscape.txt"
    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        rule(),
        f"  DRUG DISCOVERY LANDSCAPE  |  {target.upper()}",
        f"  {now_str}",
        rule(),
        "",
        "  This report covers dimensions with active or emerging published",
        "  research only. Dimensions with no recent literature are omitted.",
        "",
    ]

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — AT A GLANCE TABLE
    # ════════════════════════════════════════════════════════════════════════
    lines += [
        rule("─"),
        "  SECTION 1  |  WHAT THE FIELD IS WORKING ON  (at a glance)",
        rule("─"),
        "",
        f"  {'AREA':<26} {'DIMENSION':<32} STATUS",
        f"  {'-'*26} {'-'*32} {'-'*10}",
    ]

    for dim_key, score, status in visible:
        dim = DIMENSIONS[dim_key]
        lines.append(
            f"  {dim['group']:<26} {dim['label']:<32} {STATUS_LABEL[status]}"
        )

    lines += [
        "",
        "  ● Active   = substantial recent literature (≥2 papers post-2020)",
        "  ◑ Emerging = nascent activity (1 paper post-2020)",
        "",
    ]

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — DIMENSION DETAIL (Active/Emerging only)
    # grouped by area
    # ════════════════════════════════════════════════════════════════════════
    lines += [
        rule("─"),
        "  SECTION 2  |  DIMENSION DETAIL",
        rule("─"),
        "  For each active or emerging dimension: what the field has",
        "  established, and key supporting papers (newest first).",
        "",
    ]

    # Group visible dims by area, preserving rank order within each group
    grouped: dict[str, list[tuple]] = {}
    for item in visible:
        dim_key = item[0]
        g = DIMENSIONS[dim_key]["group"]
        grouped.setdefault(g, []).append(item)

    for group, items in grouped.items():
        lines += [
            f"  ▌ {group.upper()}",
            thin(),
        ]
        for dim_key, score, status in items:
            dim   = DIMENSIONS[dim_key]
            arts  = filtered.get(dim_key, [])
            synth = syntheses.get(dim_key, "").strip()

            lines += [
                "",
                f"  ◆ {dim['label']}  [{STATUS_LABEL[status]}]",
                f"    {dim['why_it_matters']}",
                "",
            ]

            if synth:
                lines.append("    What the field shows:")
                # wrap each sentence on its own indented line
                for sentence in synth.split(". "):
                    s = sentence.strip().rstrip(".")
                    if s:
                        lines.append(f"      • {s}.")
                lines.append("")

            # Top 5 papers, newest first
            top_papers = sorted(arts, key=_year_int, reverse=True)[:5]
            if top_papers:
                lines.append(f"    Key papers:")
                for a in top_papers:
                    lines.append(
                        f"      [{a.get('year','?')}] {a['title']}"
                    )
                    lines.append(
                        f"             PMID {a['pmid']}"
                        + (f"  |  {a['journal']}" if a.get('journal') else "")
                    )
                lines.append("")

        lines.append("")  # space between groups

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — CROSS-CUTTING PAPERS (only if any exist)
    # ════════════════════════════════════════════════════════════════════════
    # Filter cooccurrence to only visible dimensions
    visible_keys = {dim_key for dim_key, _, _ in visible}
    pmid_title   = {
        a["pmid"]: (a.get("title", ""), a.get("year", ""), a.get("journal", ""))
        for arts in filtered.values() for a in arts
    }
    visible_cooccur = {
        pmid: [d for d in dims if any(
            DIMENSIONS[k]["label"] == d for k in visible_keys
        )]
        for pmid, dims in cooccur.items()
    }
    visible_cooccur = {p: d for p, d in visible_cooccur.items() if len(d) >= 2}

    if visible_cooccur:
        lines += [
            rule("─"),
            "  SECTION 3  |  MUST-READ PAPERS  (span multiple active dimensions)",
            rule("─"),
            "  These papers sit at the intersection of several key design",
            "  factors — highest-priority reading for program planning.",
            "",
        ]
        for pmid, dims in sorted(visible_cooccur.items()):
            title, year, journal = pmid_title.get(pmid, ("", "", ""))
            lines += [
                f"  [{year}] {title}",
                f"  PMID {pmid}" + (f"  |  {journal}" if journal else ""),
                f"  Relevant to: {' | '.join(dims)}",
                "",
            ]

    lines += [rule(), f"  END  |  {target.upper()}  |  {now_str}", rule()]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Text report → {txt_path}")

    # ════════════════════════════════════════════════════════════════════════
    # JSON OUTPUT — clean, flat, PDF-pipeline friendly
    # ════════════════════════════════════════════════════════════════════════
    json_path = f"{safe}/txts/{out_prefix}{safe}_landscape.json"
    payload = {
        "target":    target,
        "generated": now_str,
        "summary": {
            "active_count":   sum(1 for _, _, s in visible if s == "Active"),
            "emerging_count": sum(1 for _, _, s in visible if s == "Emerging"),
            "total_shown":    len(visible),
        },
        "dimensions": [
            {
                "rank":          i + 1,
                "dim_key":       dim_key,
                "group":         DIMENSIONS[dim_key]["group"],
                "label":         DIMENSIONS[dim_key]["label"],
                "status":        status,
                "why_it_matters": DIMENSIONS[dim_key]["why_it_matters"],
                "synthesis":     syntheses.get(dim_key, ""),
                "top_papers": [
                    {
                        "pmid":    a["pmid"],
                        "year":    a["year"],
                        "title":   a["title"],
                        "journal": a.get("journal", ""),
                    }
                    for a in sorted(
                        filtered.get(dim_key, []), key=_year_int, reverse=True
                    )[:5]
                ],
            }
            for i, (dim_key, score, status) in enumerate(visible)
        ],
        "must_read_papers": [
            {
                "pmid":       pmid,
                "year":       pmid_title.get(pmid, ("", "", ""))[1],
                "title":      pmid_title.get(pmid, ("", "", ""))[0],
                "journal":    pmid_title.get(pmid, ("", "", ""))[2],
                "dimensions": dims,
            }
            for pmid, dims in visible_cooccur.items()
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON        → {json_path}")

def run_pipeline(
    target:        str,
    api_key:       str,
    aliases:       list[str] | None = None,
    max_per_query: int  = MAX_PER_QUERY,
    use_llm:       bool = True,
    out_prefix:    str  = "",
) -> dict[str, list[dict]]:

    print(f"\n{'='*60}")
    print(f"  Drug Discovery Landscape: {target}")
    if aliases:
        print(f"  Aliases: {', '.join(aliases)}")
    print(f"{'='*60}\n")

    # 1. Build queries
    queries = build_queries(target, aliases)
    print(f"  {len(queries)} dimensions.\n")

    # 2. Fetch from PubMed
    print("── FETCHING ─────────────────────────────────────────────────────")
    raw = fetch_all(queries, api_key)

    # 3. Stage 1 — title anchor pre-filter (no LLM)
    print("\n── STAGE 1: TITLE ANCHOR PRE-FILTER ─────────────────────────────")
    filtered = prefilter_all(raw)
    for dim_key in filtered:
        n_raw  = len(raw.get(dim_key, []))
        n_kept = len(filtered[dim_key])
        label  = DIMENSIONS[dim_key]["label"]
        print(f"  {label:<45} {n_raw:>3} → {n_kept:>3} kept")

    total_kept = sum(len(v) for v in filtered.values())
    print(f"\n  Total after pre-filter: {total_kept} articles")

    # 4. Stage 2 — LLM synthesis on clean data
    print("\n── STAGE 2: LLM SYNTHESIS ───────────────────────────────────────")
    synthesis = synthesize_all(filtered, target, use_llm=use_llm)

    # 5. Build report
    print("\n── SAVING REPORT ────────────────────────────────────────────────")
    build_report(filtered, synthesis, target, out_prefix)

    return filtered
