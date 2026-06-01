import time
import requests
from typing import Optional
import json
import re
import csv
import os
import argparse
import glob
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd



EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def build_queries(protein_name: str) -> dict[str, str]:
    """
    Returns a dict of {label: query_string} for a given protein target.
    Each query is tuned to catch a different linguistic pattern found
    in drug-discovery titles / abstracts.
    """
    p = protein_name  # shorthand

    queries = {

        # ── Pattern 1: "Discovery of X as inhibitors of PROTEIN" ──────────
        "discovery_inhibitors": (
            f'("{p}"[TIAB]) AND '
            f'("discovery"[Title] OR "identification"[Title] OR "design"[Title]) AND '
            f'("inhibitor"[TIAB] OR "inhibitors"[TIAB])'
        ),

        # ── Pattern 2: Named scaffold classes ("derivatives", "analogues") ─
        "scaffold_derivatives": (
            f'("{p}"[TIAB]) AND '
            f'("derivatives"[Title] OR "analogues"[Title] OR "analogs"[Title] '
            f' OR "series"[Title] OR "scaffold"[Title])'
        ),

        # ── Pattern 3: SMILES-adjacent / substructure language ─────────────
        "substructure_class": (
            f'("{p}"[TIAB]) AND '
            f'("pharmacophore"[TIAB] OR "scaffold"[TIAB] OR "core structure"[TIAB] '
            f' OR "substructure"[TIAB] OR "chemotype"[TIAB])'
        ),

        # ── Pattern 4: "novel class" framing ──────────────────────────────
        "novel_class": (
            f'("{p}"[TIAB]) AND '
            f'("novel class"[Title] OR "new class"[Title] OR "novel series"[Title] '
            f' OR "new scaffold"[Title])'
        ),

        # ── Pattern 5: Specific ring systems / heterocycles (broad) ────────
        "heterocyclic_inhibitors": (
            f'("{p}"[TIAB]) AND '
            f'("pyrimidine"[TIAB] OR "indole"[TIAB] OR "quinoline"[TIAB] '
            f' OR "benzimidazole"[TIAB] OR "triazole"[TIAB] OR "oxazole"[TIAB] '
            f' OR "thiazolidine"[TIAB] OR "flavonoid"[TIAB]) AND '
            f'("inhibit"[TIAB] OR "inhibitor"[TIAB])'
        ),

        # ── Pattern 6: Treatment / therapeutic framing ─────────────────────
        "treatment_therapeutic": (
            f'("{p}"[TIAB]) AND '
            f'("treatment"[Title] OR "therapeutic"[Title] OR "drug"[Title]) AND '
            f'("compound"[TIAB] OR "molecule"[TIAB] OR "agent"[TIAB])'
        ),

        # ── Pattern 7: Structure–activity relationship papers ───────────────
        "SAR_QSAR": (
            f'("{p}"[TIAB]) AND '
            f'("structure-activity"[TIAB] OR "SAR"[TIAB] OR "QSAR"[TIAB] '
            f' OR "molecular docking"[TIAB] OR "binding affinity"[TIAB])'
        ),

        # ── Pattern 8: Natural product / plant-derived leads ────────────────
        "natural_product_leads": (
            f'("{p}"[TIAB]) AND '
            f'("natural product"[TIAB] OR "plant-derived"[TIAB] '
            f' OR "phytochemical"[TIAB] OR "alkaloid"[TIAB] OR "terpenoid"[TIAB])'
        ),

        # ── Pattern 9: Fragment-based / virtual screening ───────────────────
        "fragment_virtual_screening": (
            f'("{p}"[TIAB]) AND '
            f'("fragment-based"[TIAB] OR "virtual screening"[TIAB] '
            f' OR "high-throughput screening"[TIAB] OR "HTS"[TIAB] '
            f' OR "hit compound"[TIAB] OR "lead compound"[TIAB])'
        ),

        # ── Pattern 10: Allosteric / covalent / PROTAC modalities ──────────
        "modality_specific": (
            f'("{p}"[TIAB]) AND '
            f'("allosteric"[TIAB] OR "covalent inhibitor"[TIAB] '
            f' OR "PROTAC"[TIAB] OR "degrader"[TIAB] OR "bifunctional"[TIAB])'
        ),
    }

    return queries


def esearch(
    query: str,
    api_key: str,
    max_results: int = 50,
    db: str = "pubmed",
) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": db,
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "api_key": api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def efetch_abstracts(
    pmids: list[str],
    api_key: str,
    db: str = "pubmed",
    batch_size: int = 20,
) -> list[dict]:
    """
    Fetch title + abstract for a list of PMIDs.
    Returns a list of dicts: {pmid, title, abstract, authors, journal, year}
    """
    results = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        url = f"{EUTILS_BASE}/efetch.fcgi"
        params = {
            "db": db,
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
            "api_key": api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        articles = _parse_pubmed_xml(resp.text)
        results.extend(articles)

        # NCBI rate limit: 10 req/s with API key, 3/s without
        time.sleep(0.15)

    return results


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML (efetch abstract format) into dicts."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        title_elem = article_elem.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()) if title_elem is not None else ""

        # Abstract can have multiple AbstractText sections (structured abstracts)
        abstract_parts = article_elem.findall(".//AbstractText")
        if abstract_parts:
            abstract = " ".join(
                "".join(p.itertext()) for p in abstract_parts
            )
        else:
            abstract = ""

        # Authors
        author_elems = article_elem.findall(".//Author")
        authors = []
        for ae in author_elems:
            last = ae.findtext("LastName", "")
            fore = ae.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())

        journal = article_elem.findtext(".//Journal/Title", "")
        year    = article_elem.findtext(".//PubDate/Year", "")

        articles.append({
            "pmid":     pmid,
            "title":    title.strip(),
            "abstract": abstract.strip(),
            "authors":  authors,
            "journal":  journal,
            "year":     year,
        })

    return articles


def fetch_for_queries(
    queries: dict[str, str],
    api_key: str,
    max_per_query: int = 50,
) -> dict[str, list[dict]]:
    """
    Run all queries and return {query_label: [article_dicts]}.
    Deduplicates across queries by PMID.
    """
    seen_pmids: set[str] = set()
    results: dict[str, list[dict]] = {}

    for label, query in queries.items():
        print(f"  Searching [{label}] ...", end=" ", flush=True)
        pmids = esearch(query, api_key, max_results=max_per_query)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)

        if new_pmids:
            articles = efetch_abstracts(new_pmids, api_key)
            results[label] = articles
            print(f"{len(articles)} new articles")
        else:
            results[label] = []
            print("0 new articles (all duplicates)")

        time.sleep(0.35)  # be polite between query bursts

    return results

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

# ── Prompt is intentionally minimal for a 7B model ───────────────
# Shorter context = faster inference + fewer hallucinations
SCREEN_SYSTEM = """\
You are a medicinal chemistry expert screening paper titles.
Respond ONLY with a valid JSON object, no explanation, no markdown.
Schema: {"relevant": true/false, "scaffold": "name or empty string", "reason": "one short phrase"}

A title is relevant if it announces or names a specific chemical scaffold,
compound class, pharmacophore, heterocycle, natural product derivative,
or substructure as a potential inhibitor or therapeutic agent against a protein target.

Examples of RELEVANT titles:
- "Discovery of 2,3-diindolylmethanes as novel EGFR inhibitors"   → relevant, scaffold="diindolylmethane"
- "Synthesis and evaluation of quinazoline derivatives as VEGFR-2 inhibitors" → relevant, scaffold="quinazoline"
- "Indole-based compounds as potent CDK2 inhibitors"              → relevant, scaffold="indole"

Examples of NOT RELEVANT titles:
- "EGFR signaling pathway in lung cancer progression"             → not relevant
- "Computational study of protein-ligand binding mechanisms"      → not relevant
- "Clinical outcomes of EGFR-mutant NSCLC patients"              → not relevant
"""

def screen_title(title: str, timeout: int = 30) -> dict:
    """
    Returns {"relevant": bool, "scaffold": str, "reason": str}
    Falls back to {"relevant": False, "scaffold": "", "reason": "parse_error"} on failure.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "system": SCREEN_SYSTEM,
        "prompt": f"Screen this title: {title}",
        "stream": False,
        "options": {
            "temperature": 0.05,   # near-deterministic
            "num_predict": 80,     # titles → short JSON output is enough
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Strip markdown fences just in case
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)

    except json.JSONDecodeError:
        return {"relevant": False, "scaffold": "", "reason": "parse_error"}
    except requests.RequestException as e:
        return {"relevant": False, "scaffold": "", "reason": f"request_error: {e}"}


def screen_all_titles(
    articles: list[dict],
    output_dir: str = "output",
    protein: str = "protein",
) -> tuple[list[dict], list[dict]]:
    """
    Screen all article titles.
    Returns (relevant_articles, all_screened_articles).
    Each article gets a 'screen' key with the LLM result.
    """
   
    screened   = []
    relevant   = []
    parse_errs = 0

    print(f"\nScreening {len(articles)} titles ...\n")

    for i, art in enumerate(articles):
        title  = art.get("title", "")
        pmid   = art.get("pmid", "?")
        result = screen_title(title)

        art_screened = {**art, "screen": result}
        screened.append(art_screened)

        flag = "✓" if result.get("relevant") else "–"
        scaffold_hint = f'  [{result.get("scaffold","")}]' if result.get("relevant") else ""
        print(f"  {flag} [{i+1:>3}/{len(articles)}] PMID {pmid}{scaffold_hint}")
        print(f"       {title[:90]}")

        if result.get("relevant"):
            relevant.append(art_screened)

        if result.get("reason") == "parse_error":
            parse_errs += 1

    # ── Save CSV of ALL screened titles ──────────────────────────
    csv_path = os.path.join(output_dir,"scaffolds", f"{protein}_title_screen.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pmid", "year", "journal", "title",
            "relevant", "scaffold", "reason", "source_query"
        ])  # default delimiter is ","
        writer.writeheader()
        for art in screened:
            s = art.get("screen", {})
            writer.writerow({
                "pmid":         art.get("pmid", ""),
                "year":         art.get("year", ""),
                "journal":      art.get("journal", ""),
                "title":        art.get("title", ""),
                "relevant":     s.get("relevant", False),
                "scaffold":     s.get("scaffold", ""),
                "reason":       s.get("reason", ""),
                "source_query": art.get("source_query", ""),
            })
    print(f"\n  Screening CSV saved → {csv_path}")

    # ── Save relevant-only JSON (feed into Block 3 abstract extraction) ──
    rel_path = os.path.join(output_dir,"scaffolds", f"{protein}_relevant_titles.json")
    with open(rel_path, "w", encoding="utf-8") as f:
        json.dump(relevant, f, indent=2, ensure_ascii=False)
    print(f"  Relevant articles JSON saved → {rel_path}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Total screened : {len(screened)}")
    print(f"  Relevant       : {len(relevant)}  ({len(relevant)/len(screened)*100:.1f}%)")
    print(f"  Parse errors   : {parse_errs}")
    print(f"{'='*50}\n")

    return relevant, csv_path


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# ── PubChem helpers ───────────────────────────────────────────────

def get_pubchem_cid(name: str) -> int | None:
    encoded = urllib.parse.quote(name)
    url = f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
    except Exception:
        pass
    return None


def get_smiles_from_cid(cid: int) -> str | None:
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/SMILES/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("SMILES")
    except Exception:
        pass
    return None


def get_iupac_name(cid: int) -> str | None:
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/IUPACName/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IUPACName")
    except Exception:
        pass
    return None



def get_smiles_from_cactus(name: str) -> str | None:
    """Fallback: NCI CACTUS Chemical Identifier Resolver."""
    encoded = urllib.parse.quote(name)
    url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/smiles"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.text.strip():
            smiles = r.text.strip()
            if len(smiles) < 300 and not smiles.lower().startswith(("page", "<", "error")):
                return smiles
    except Exception:
        pass
    return None


def _simplify_name(name: str) -> str:
    """Strip generic chemistry suffixes before retrying lookup."""
    return re.sub(
        r"\b(derivative|derivatives|analog|analogue|compound|class|based|scaffold|core)\b",
        "", name, flags=re.IGNORECASE
    ).strip(" -,")


def fetch_scaffold_data(name: str) -> dict:
    """
    Try PubChem first, fall back to CACTUS for SMILES.
    Returns a dict. 'smiles' will be None if nothing was found.
    """
    result = dict(cid=None, smiles=None, iupac=None, img_url=None,
                  pubchem_url=None)

    # ── PubChem ───────────────────────────────────────────────────
    cid = get_pubchem_cid(name) or get_pubchem_cid(_simplify_name(name))

    if cid:
        smiles = get_smiles_from_cid(cid)
        if not smiles:                       # retry once on transient failure
            time.sleep(0.5)
            smiles = get_smiles_from_cid(cid)
        result.update(
            cid        = cid,
            smiles     = smiles,
            iupac      = get_iupac_name(cid),
            # img_url    = get_structure_png_url(cid),
            pubchem_url= f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        )
        time.sleep(0.22)
        return result

    # ── CACTUS fallback (SMILES only, no image) ───────────────────
    smiles = get_smiles_from_cactus(name) or get_smiles_from_cactus(_simplify_name(name))
    if smiles:
        result["smiles"] = smiles
    time.sleep(0.15)
    return result


# ── CSV processing ────────────────────────────────────────────────

def load_and_filter_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=",", dtype=str).fillna("")

    if "scaffold" not in df.columns:
        raise ValueError(
            f"'scaffold' column not found. Available: {list(df.columns)}"
        )

    before = len(df)
    df = df[df["scaffold"].str.strip().str.lower().replace("empty string", "") != ""]
    df = df[df.get("relevant", pd.Series(["true"] * len(df))).str.strip().str.lower() == "true"]
    print(f"  {before} rows → {len(df)} with non-empty scaffold + relevant=true")
    return df.reset_index(drop=True)


def deduplicate_scaffolds(df: pd.DataFrame) -> list[dict]:
    """One entry per unique scaffold name, collecting all articles."""
    grouped: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = row["scaffold"].strip().lower()
        if key not in grouped:
            grouped[key] = {"scaffold_name": row["scaffold"].strip(), "articles": []}
        grouped[key]["articles"].append({
            "pmid":    row.get("pmid", ""),
            "title":   row.get("title", ""),
            "year":    row.get("year", ""),
            "journal": row.get("journal", ""),
        })
    return list(grouped.values())


# ── CSV save ──────────────────────────────────────────────────────

def save_enriched_csv(enriched: list[dict], out_path: str) -> None:
    """
    Flat CSV — one row per (scaffold × article).
    Only rows where SMILES is not empty are written.
    """
    rows = []

    for i, s in enumerate(enriched):
        # print(s)
        if not s.get("smiles"):          # ← SMILES-only filter
            continue
        for art in s["articles"]:
            rows.append({
                "scaffold_index": i,
                "scaffold":       s["scaffold_name"],
                "smiles":         s["smiles"],
                "cid":            s.get("cid") or "",
                "iupac":          s.get("iupac") or "",
                # "img_url":        s.get("img_url") or "",
                "pubchem_url":    s.get("pubchem_url") or "",
                "pmid":           art["pmid"],
                "title":          art["title"],
                "year":           art["year"],
                "journal":        art["journal"],
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    unique = len({r["scaffold"] for r in rows})
    print(f"  Saved enriched CSV: {unique} scaffolds with SMILES, "
          f"{len(rows)} rows → {out_path}")
    

import argparse
import glob
import json
import base64
from datetime import datetime
from pathlib import Path
from io import BytesIO

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw


# ── SMILES → base64 image ─────────────────────────────────────────
def smiles_to_base64(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        Chem.rdDepictor.Compute2DCoords(mol)

        img = Draw.MolToImage(
            mol,
            size=(320, 200),
            kekulize=True
        )

        buf = BytesIO()
        img.save(buf, format="PNG")

        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception:
        return ""


# ── Load CSV ──────────────────────────────────────────────────────
def load_enriched_csv(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = df[df["smiles"].str.strip() != ""]

    grouped = {}

    for _, row in df.iterrows():
        key = row["scaffold"].strip().lower()
        smiles = row["smiles"].strip()

        # ✅ ALWAYS use RDKit (ignore PubChem)
        img_url = smiles_to_base64(smiles)

        if key not in grouped:
            grouped[key] = {
                "scaffold": row["scaffold"].strip(),
                "smiles": smiles,
                "cid": row.get("cid", "").strip(),
                "iupac": row.get("iupac", "").strip(),
                "img_url": img_url,
                "pubchem_url": row.get("pubchem_url", "").strip(),
                "articles": [],
            }

        grouped[key]["articles"].append({
            "pmid": row.get("pmid", "").strip(),
            "title": row.get("title", "").strip(),
            "year": row.get("year", "").strip(),
            "journal": row.get("journal", "").strip(),
        })

    return list(grouped.values())


# ── HTML ──────────────────────────────────────────────────────────
def build_html(scaffolds, protein, csv_filename):
    data_json = json.dumps(scaffolds, ensure_ascii=False)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{protein} Scaffold Picker</title>

<style>
body {{ font-family: Arial; background:#f5f5f5; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,320px); gap:14px; padding:20px; }}
.card {{ background:white; border:1px solid #ccc; padding:10px; cursor:pointer; }}
.card.selected {{ border:2px solid green; }}
img {{ width:100%; height:180px; object-fit:contain; }}
.smiles {{ font-size:11px; color:#555; word-break:break-all; }}
.art {{ margin-top:6px; border-left:2px solid #ccc; padding-left:6px; }}
</style>
</head>

<body>

<h2>{protein} scaffold picker</h2>
<p>{len(scaffolds)} scaffolds · {ts}</p>

<input id="search" placeholder="search..." oninput="filter()">
<button onclick="saveCSV()">save selected</button>
<span id="count">0 selected</span>

<div class="grid" id="grid"></div>

<script>
const DATA = {data_json};
const selected = new Set();

function build() {{
  const grid = document.getElementById('grid');

  DATA.forEach((s,i)=>{{
    const card = document.createElement('div');
    card.className = 'card';
    card.id = 'c'+i;

    let arts = s.articles.slice(0,2).map(a => `
      <div class="art">
        <a href="https://pubmed.ncbi.nlm.nih.gov/${{a.pmid}}" target="_blank">
          ${{a.title || '(no title)'}}
        </a><br>
        <small>${{a.year || ''}} ${{a.journal || ''}}</small>
      </div>
    `).join('');

    let more = '';
    if (s.articles.length > 2) {{
      more = `<div id="more${{i}}" style="display:none;">
        ${{
          s.articles.slice(2).map(a => `
            <div class="art">
              <a href="https://pubmed.ncbi.nlm.nih.gov/${{a.pmid}}" target="_blank">
                ${{a.title || '(no title)'}}
              </a>
              <small>${{a.year || ''}}</small>
            </div>
          `).join('')
        }}
      </div>
      <button onclick="event.stopPropagation();toggle(${{i}})">
        +${{s.articles.length-2}} more
      </button>`;
    }}

    card.innerHTML = `
      <img src="${{s.img_url}}">
      <b>${{s.scaffold}}</b>
      <div class="smiles">${{s.smiles}}</div>
      ${{arts}}
      ${{more}}
    `;

    card.onclick = () => select(i);
    grid.appendChild(card);
  }});
}}

function select(i) {{
  const c = document.getElementById('c'+i);
  if(selected.has(i)) {{
    selected.delete(i);
    c.classList.remove('selected');
  }} else {{
    selected.add(i);
    c.classList.add('selected');
  }}
  document.getElementById('count').innerText = selected.size + " selected";
}}

function toggle(i) {{
  const el = document.getElementById('more'+i);
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}

function filter() {{
  const q = document.getElementById('search').value.toLowerCase();
  DATA.forEach((s,i)=>{{
    const c = document.getElementById('c'+i);
    const txt = (s.scaffold + s.smiles + s.articles.map(a=>a.title).join(' ')).toLowerCase();
    c.style.display = txt.includes(q) ? '' : 'none';
  }});
}}

function saveCSV() {{
  if(!selected.size) return alert("nothing selected");

  const cols = ['scaffold','smiles','cid','iupac','pubchem_url','pmid','title','year','journal'];
  let rows = [cols.join(',')];

  [...selected].forEach(i => {{
    const s = DATA[i];
    s.articles.forEach(a => {{
      rows.push(cols.map(c => {{
        const v = c==='scaffold'?s.scaffold:
                  c==='smiles'?s.smiles:
                  c==='cid'?s.cid:
                  c==='iupac'?s.iupac:
                  c==='pubchem_url'?s.pubchem_url:
                  a[c] || '';
        return '"' + String(v).replace(/"/g,'""') + '"';
      }}).join(','));
    }});
  }});

  const blob = new Blob([rows.join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = "{protein}_selected.csv";
  a.click();
}}

build();
</script>

</body>
</html>
"""


import tkinter as tk
from tkinter import ttk, messagebox
import io
import webbrowser
from PIL import Image, ImageTk
 
# RDKit imports
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
 
 
# ── Molecule rendering ────────────────────────────────────────────────────────
 
def smiles_to_photoimage(smiles: str, size: int = 160) -> ImageTk.PhotoImage | None:
    """Render a SMILES string to a Tk PhotoImage via RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 1.6
        opts.padding = 0.12
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        bg = Image.new("RGBA", img.size, "white")
        bg.paste(img, mask=img.split()[3])
        bg = bg.convert("RGB")
        buf = io.BytesIO()
        bg.save(buf, format="PNG")
        buf.seek(0)
        return ImageTk.PhotoImage(Image.open(buf))
    except Exception:
        return None
 
 
# ── Clickable article link label ──────────────────────────────────────────────
 
class ArticleLink(tk.Label):
    """A label that looks like a hyperlink and opens PubMed on click."""
 
    def __init__(self, parent, title: str, year: str, pmid: str, bg: str, **kw):
        display = f"[{year}] {title}" if year else title
        if len(display) > 80:
            display = display[:77] + "…"
        super().__init__(parent, text=display,
                         font=("Courier New", 10, "underline"),
                         fg="#2471a3", bg=bg,
                         cursor="hand2",
                         wraplength=210, justify="left",
                         anchor="w", **kw)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        if url:
            self.bind("<Button-1>", lambda _: webbrowser.open(url))
            self.bind("<Enter>", lambda _: self.configure(fg="#1a5276"))
            self.bind("<Leave>", lambda _: self.configure(fg="#2471a3"))
        else:
            self.configure(fg="#888", cursor="arrow",
                           font=("Courier New", 10))
 
 
# ── Scaffold card ─────────────────────────────────────────────────────────────
 
class ScaffoldCard(tk.Frame):
    IDLE_BG = "#ffffff"
    SEL_BG  = "#eaf4fb"
    SEL_BD  = "#2980b9"
    IDLE_BD = "#dcdad4"
 
    def __init__(self, parent, scaffold: dict, var: tk.BooleanVar,
                 img_size: int = 160, **kw):
        super().__init__(parent, bg=self.IDLE_BG, relief="flat",
                         highlightthickness=2,
                         highlightbackground=self.IDLE_BD,
                         cursor="hand2", **kw)
        self._var = var
        self._photo = None
 
        var.trace_add("write", self._sync_style)
        self.bind("<Button-1>", self._toggle)
        self._build(scaffold, img_size)
        self._sync_style()
 
    def _build(self, s: dict, size: int):
        bg = self.IDLE_BG
 
        # ── top row: checkbox + year badge ───────────────────────────────────
        top = tk.Frame(self, bg=bg)
        top.pack(fill="x", padx=6, pady=(6, 0))
 
        self._cb = tk.Checkbutton(top, variable=self._var,
                                  bg=bg, activebackground=bg, cursor="hand2")
        self._cb.pack(side="left")
 
        years = []
        for art in s.get("articles", []):
            try:
                years.append(int(art["year"]))
            except (KeyError, ValueError, TypeError):
                pass
        if years:
            badge = tk.Label(top, text=f"latest {max(years)}",
                             font=("Courier New", 10, "bold"),
                             bg="#2980b9", fg="white", padx=5, pady=2)
            badge.pack(side="right", padx=2)
            badge.bind("<Button-1>", self._toggle)
 
        # ── molecule image ────────────────────────────────────────────────────
        smiles = s.get("smiles", "")
        self._photo = smiles_to_photoimage(smiles, size) if smiles else None
 
        if self._photo:
            img_lbl = tk.Label(self, image=self._photo, bg=bg, bd=0)
            img_lbl.image = self._photo
            img_lbl.pack(pady=(4, 0))
            img_lbl.bind("<Button-1>", self._toggle)
        else:
            ph = tk.Label(self, text="no structure",
                          width=20, height=5, bg="#f0ede6",
                          fg="#bbb", font=("Courier New", 9))
            ph.pack(pady=(4, 0))
            ph.bind("<Button-1>", self._toggle)
 
        # ── separator ─────────────────────────────────────────────────────────
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=6)
 
        # ── scaffold name ─────────────────────────────────────────────────────
        name_lbl = tk.Label(self,
                            text=(s.get("scaffold") or "Unknown"),
                            font=("Courier New", 9, "bold"),
                            bg=bg, fg="#111",
                            wraplength=size + 20, justify="center")
        name_lbl.pack(padx=8)
        name_lbl.bind("<Button-1>", self._toggle)
 
        # ── SMILES ────────────────────────────────────────────────────────────
        if smiles:
            display_smiles = smiles if len(smiles) <= 44 else smiles[:41] + "…"
            sm_lbl = tk.Label(self, text=display_smiles,
                              font=("Courier New", 9),
                              bg=bg, fg="#888",
                              wraplength=size + 20, justify="center")
            sm_lbl.pack(pady=(2, 0), padx=8)
            sm_lbl.bind("<Button-1>", self._toggle)
 
        # ── CID / IUPAC ───────────────────────────────────────────────────────
        cid = s.get("cid", "")
        iupac = s.get("iupac", "")
        meta_parts = []
        if cid:
            meta_parts.append(f"CID {cid}")
        if iupac and iupac != s.get("scaffold"):
            meta_parts.append(iupac)
        if meta_parts:
            meta_lbl = tk.Label(self, text=" · ".join(meta_parts),
                                font=("Courier New", 7),
                                bg=bg, fg="#aaa",
                                wraplength=size + 20, justify="center")
            meta_lbl.pack(pady=(1, 0), padx=8)
            meta_lbl.bind("<Button-1>", self._toggle)
 
        # ── articles section ──────────────────────────────────────────────────
        articles = s.get("articles", [])
        if articles:
            ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=(6, 4))
 
            art_header = tk.Label(self,
                                  text=f"📄  {len(articles)} article{'s' if len(articles) != 1 else ''}",
                                  font=("Courier New", 8, "bold"),
                                  bg=bg, fg="#555", anchor="w")
            art_header.pack(fill="x", padx=10)
            art_header.bind("<Button-1>", self._toggle)
 
            for art in articles:
                pmid    = art.get("pmid", "")
                title   = art.get("title", "No title")
                year    = str(art.get("year", ""))
                journal = art.get("journal", "")
 
                link = ArticleLink(self, title=title, year=year,
                                   pmid=pmid, bg=bg)
                link.pack(fill="x", padx=10, pady=(2, 0))
 
                if journal:
                    jlbl = tk.Label(self, text=journal,
                                    font=("Courier New", 6, "italic"),
                                    bg=bg, fg="#bbb",
                                    wraplength=size + 20,
                                    justify="left", anchor="w")
                    jlbl.pack(fill="x", padx=10, pady=(0, 2))
                    jlbl.bind("<Button-1>", self._toggle)
 
        # bottom padding
        tk.Label(self, text="", bg=bg).pack()
 
    def _toggle(self, *_):
        self._var.set(not self._var.get())
 
    def _sync_style(self, *_):
        sel = self._var.get()
        bg  = self.SEL_BG if sel else self.IDLE_BG
        bd  = self.SEL_BD if sel else self.IDLE_BD
        self.configure(bg=bg, highlightbackground=bd)
        self._recolor(self, bg)
 
    def _recolor(self, widget, bg: str):
        try:
            if not isinstance(widget, (tk.Checkbutton, ttk.Separator, ArticleLink)):
                widget.configure(bg=bg)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._recolor(child, bg)
 
 
# ── Main selector ─────────────────────────────────────────────────────────────
 
def show_scaffold_selector(TARGET_PROTEIN: str,
                           scaffolds: list[dict],
                           img_size: int = 160,
                           cols: int = 4) -> list[dict]:
    """
    Opens a Tk window to browse and select scaffolds.
    Each card shows: RDKit molecule image, scaffold name, SMILES,
    CID/IUPAC (if present), and clickable PubMed article links.
    Returns the list of selected scaffold dicts.
    """
    root = tk.Tk()
    root.title(f"Scaffold Selector — {TARGET_PROTEIN}")
    root.geometry("1280x860")
    root.minsize(700, 500)
    root.configure(bg="#f5f3ee")
 
    # ── Header ────────────────────────────────────────────────────────────────
    header = tk.Frame(root, bg="#12121f", pady=14)
    header.pack(fill="x")
    tk.Label(header, text="⬡  Scaffold Selector",
             font=("Courier New", 15, "bold"),
             bg="#12121f", fg="#e8e4dc").pack(side="left", padx=20)
    tk.Label(header, text=f"target: {TARGET_PROTEIN}",
             font=("Courier New", 10),
             bg="#12121f", fg="#6c7a89").pack(side="left", padx=4)
    count_var = tk.StringVar(value="0 selected")
    tk.Label(header, textvariable=count_var,
             font=("Courier New", 11, "bold"),
             bg="#12121f", fg="#2ecc71").pack(side="right", padx=20)
 
    # ── Search bar ────────────────────────────────────────────────────────────
    sf = tk.Frame(root, bg="#f5f3ee", pady=8)
    sf.pack(fill="x", padx=20)
    tk.Label(sf, text="Filter:", font=("Courier New", 10),
             bg="#f5f3ee", fg="#555").pack(side="left")
    search_var = tk.StringVar()
    tk.Entry(sf, textvariable=search_var, font=("Courier New", 10),
             width=44, relief="flat", bd=4, bg="#fff").pack(side="left", padx=8)
    tk.Label(sf, text="(name · SMILES · article title · journal)",
             font=("Courier New", 8), bg="#f5f3ee", fg="#bbb").pack(side="left")
 
    # ── Scrollable canvas ─────────────────────────────────────────────────────
    outer = tk.Frame(root, bg="#f5f3ee")
    outer.pack(fill="both", expand=True, padx=20, pady=(0, 8))
 
    canvas = tk.Canvas(outer, bg="#f5f3ee", highlightthickness=0)
    vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
 
    inner = tk.Frame(canvas, bg="#f5f3ee")
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>",
                lambda e: canvas.itemconfig(win_id, width=e.width))
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
 
    # ── Build cards ───────────────────────────────────────────────────────────
    check_vars: list[tk.BooleanVar] = []
    cards: list[tuple[ScaffoldCard, dict]] = []
 
    def update_count(*_):
        n = sum(v.get() for v in check_vars)
        count_var.set(f"{n} selected")
 
    for s in scaffolds:
        var = tk.BooleanVar(value=False)
        var.trace_add("write", update_count)
        check_vars.append(var)
        card = ScaffoldCard(inner, s, var, img_size=img_size)
        cards.append((card, s))
 
    def render_grid(visible):
        for card, _ in cards:
            card.grid_forget()
        for i, (card, _) in enumerate(visible):
            card.grid(row=i // cols, column=i % cols,
                      padx=6, pady=6, sticky="n")
 
    render_grid(cards)
 
    # ── Search logic ──────────────────────────────────────────────────────────
    def _article_text(s: dict) -> str:
        parts = []
        for a in s.get("articles", []):
            parts.append((a.get("title") or "").lower())
            parts.append((a.get("journal") or "").lower())
        return " ".join(parts)
 
    def on_search(*_):
        q = search_var.get().lower().strip()
        if not q:
            render_grid(cards)
            return
        visible = [
            (c, s) for c, s in cards
            if q in (s.get("scaffold") or "").lower()
            or q in (s.get("smiles") or "").lower()
            or q in (s.get("iupac") or "").lower()
            or q in _article_text(s)
        ]
        render_grid(visible)
 
    search_var.trace_add("write", on_search)
 
    # ── Bottom bar ────────────────────────────────────────────────────────────
    btn_bar = tk.Frame(root, bg="#12121f", pady=10)
    btn_bar.pack(fill="x")
 
    result_holder: list = [None]
 
    def on_confirm():
        selected = [s for (_, s), v in zip(cards, check_vars) if v.get()]
        if not selected:
            messagebox.showwarning("No selection",
                                   "Please select at least one scaffold.",
                                   parent=root)
            return
        result_holder[0] = selected
        root.destroy()
 
    def on_cancel():
        result_holder[0] = []
        root.destroy()
 
    _B = dict(font=("Courier New", 11, "bold"), relief="flat",
              padx=18, pady=6, cursor="hand2")
 
    tk.Button(btn_bar, text="✓  Confirm", bg="#27ae60", fg="white",
              activebackground="#2ecc71", command=on_confirm, **_B
              ).pack(side="left", padx=14)
    tk.Button(btn_bar, text="✕  Cancel", bg="#c0392b", fg="white",
              activebackground="#e74c3c", command=on_cancel, **_B
              ).pack(side="left")
    tk.Button(btn_bar, text="☑  Select all", bg="#2c3e50", fg="#bdc3c7",
              activebackground="#34495e",
              command=lambda: [v.set(True) for v in check_vars], **_B
              ).pack(side="right", padx=14)
    tk.Button(btn_bar, text="☐  Clear", bg="#2c3e50", fg="#bdc3c7",
              activebackground="#34495e",
              command=lambda: [v.set(False) for v in check_vars], **_B
              ).pack(side="right")
 
    root.mainloop()
    return result_holder[0] if result_holder[0] is not None else []




from collections import defaultdict


# ── CONFIG ────────────────────────────────────────────────────────

MAX_PER_QUERY   = 40                    # articles per query

# ─────────────────────────────────────────────────────────────────

def flatten_articles(query_results: dict[str, list[dict]]) -> list[dict]:
    """Merge all query buckets, tag each article with which query found it."""
    seen, flat = set(), []
    for label, articles in query_results.items():
        for art in articles:
            pmid = art["pmid"]
            if pmid not in seen:
                seen.add(pmid)
                flat.append({**art, "source_query": label})
    return flat


def save_full_json(enriched: list[dict], protein: str):
    path = os.path.join(protein,"scaffolds", f"{protein}_full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"  ✓  Full JSON  → {path}")
    return path


def save_scaffolds_csv(enriched: list[dict], protein: str):
    """Flat CSV: one row per extracted scaffold."""
    path = os.path.join(protein,"scaffolds", f"{protein}_scaffolds.csv")
    rows = []

    for art in enriched:
        ext = art.get("extraction") or {}
        for scaffold in ext.get("scaffolds", []):
            rows.append({
                "pmid":          art.get("pmid", ""),
                "year":          art.get("year", ""),
                "journal":       art.get("journal", ""),
                "title":         art.get("title", ""),
                "scaffold_name": scaffold.get("name", ""),
                "scaffold_type": scaffold.get("type", ""),
                "confidence":    scaffold.get("confidence", ""),
                "context":       scaffold.get("context", ""),
                "source_query":  art.get("source_query", ""),
            })

    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✓  Scaffolds CSV ({len(rows)} rows) → {path}")
    else:
        print("  ⚠  No scaffolds extracted — CSV not written.")

    return path


def save_scaffold_summary(enriched: list[dict], protein: str):
    """Frequency table: how often each scaffold name appears."""
    freq = defaultdict(int)
    for art in enriched:
        ext = art.get("extraction") or {}
        for s in ext.get("scaffolds", []):
            name = s.get("name", "").strip()
            if name:
                freq[name] += 1

    path = os.path.join(protein,"scaffolds", f"{protein}_scaffold_freq.csv")
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["scaffold_name", "article_count"])
        writer.writerows(sorted_freq)

    print(f"  ✓  Frequency table ({len(sorted_freq)} unique scaffolds) → {path}")
    return path


def save_scaffold_summary_txt(scaffolds_sorted: list[dict], protein: str, output_dir: str):
    """
    Saves a txt summary of scaffolds by year.
    """
    # Count by year
    year_counts = defaultdict(int)
    no_year = 0
    for scaffold in scaffolds_sorted:
        years = []
        for art in scaffold.get("articles", []):
            if art.get("year"):
                try:
                    years.append(int(art["year"]))
                except ValueError:
                    pass
        if years:
            year_counts[max(years)] += 1
        else:
            no_year += 1

    lines = []
    lines.append(f"SCAFFOLD SUMMARY: {protein}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total scaffolds identified : {len(scaffolds_sorted)}")
    lines.append(f"Scaffolds with SMILES      : {sum(1 for s in scaffolds_sorted if s.get('smiles'))}")
    lines.append("")
    lines.append("SCAFFOLDS BY YEAR (most recent publication per scaffold)")
    lines.append("-" * 40)

    for year in sorted(year_counts.keys(), reverse=True):
        lines.append(f"  {year} : {year_counts[year]}")
    if no_year:
        lines.append(f"  N/A  : {no_year}")

    lines.append("")
    lines.append("SCAFFOLD NAMES")
    lines.append("-" * 40)
    for scaffold in scaffolds_sorted:
        name = scaffold.get("scaffold_name") or scaffold.get("scaffold", "Unknown")
        years = []
        for art in scaffold.get("articles", []):
            if art.get("year"):
                try:
                    years.append(int(art["year"]))
                except ValueError:
                    pass
        year_str = str(max(years)) if years else "N/A"
        lines.append(f"  {year_str}  |  {name}")

    out_path = os.path.join(output_dir,"scaffolds" f"{protein}_scaffold_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✓  Scaffold summary → {out_path}")
    return out_path

def run_scaffold_mining(TARGET_PROTEIN):
    
    # ── Step 1: Build queries ──────────────────────────────────────
    print("Step 1/3 — Building queries ...")
    queries = build_queries(TARGET_PROTEIN)
    print(f"  {len(queries)} query templates ready.\n")

    # ── Step 2: Fetch articles ─────────────────────────────────────
    print("Step 2/3 — Fetching from PubMed ...")
    raw_results = fetch_for_queries(queries, NCBI_API_KEY, max_per_query=MAX_PER_QUERY)
    flat_articles = flatten_articles(raw_results)
    print(f"\n  Total unique articles: {len(flat_articles)}\n")

    # Optional: save raw articles before LLM step (checkpoint)
    raw_path = os.path.join(TARGET_PROTEIN, "scaffolds", f"{TARGET_PROTEIN}_raw_articles.json")
    with open(raw_path, "w") as f:
        json.dump(flat_articles, f, indent=2)
    print(f"  Raw checkpoint saved → {raw_path}\n")

    # ── Step 3: LLM extraction ─────────────────────────────────────
    print("Step 3a/4 — Screening titles with local LLM ...")
    relevant_articles, csv_path = screen_all_titles(
        flat_articles,
        output_dir=TARGET_PROTEIN,
        protein=TARGET_PROTEIN,
    )
    print(f"  {len(relevant_articles)} articles passed title screen.\n")
    
    df = load_and_filter_csv(csv_path)

    print("\n — Deduplicating scaffolds ...")
    scaffolds = deduplicate_scaffolds(df)
    print(f"  {len(scaffolds)} unique scaffold names")

    print("\n — Fetching from PubChem / CACTUS ...")
    enriched = []
    for i, s in enumerate(scaffolds):
        print(f"  [{i+1:>3}/{len(scaffolds)}] {s['scaffold_name'][:55]:<55}", end=" ")
        data = fetch_scaffold_data(s["scaffold_name"])
        enriched.append({**s, **data})
        print(f"✓ CID {data['cid']}" if data.get("cid")
              else ("✓ SMILES (CACTUS)" if data.get("smiles") else "– not found"))

    print(enriched)
   
    enriched_csv_path = str(f"{TARGET_PROTEIN}/scaffolds/{TARGET_PROTEIN}_scaffolds_enriched.csv")
    save_enriched_csv(enriched, enriched_csv_path)

    found   = sum(1 for s in enriched if s.get("smiles"))
    missing = len(enriched) - found
    print(f"\n  With SMILES  : {found}")
    print(f"  Without SMILES (excluded): {missing}")

    # Load and sort scaffolds by year (highest to lowest)
    scaffolds = load_enriched_csv(enriched_csv_path)
    print("Loaded:", len(scaffolds))

    # Sort by latest year (highest first)
    def get_max_year(scaffold):
        years = []
        for art in scaffold.get("articles", []):
            if art.get("year"):
                try:
                    years.append(int(art["year"]))
                except ValueError:
                    pass
        return max(years) if years else 0

    scaffolds_sorted = sorted(scaffolds, key=get_max_year, reverse=True)
    print("Sorted by year (highest to lowest)")
    for item in scaffolds_sorted:
        if "scaffold" in item:
            item["scaffold_name"] = item["scaffold"]

    # Save sorted scaffolds to CSV
    sorted_csv_path = os.path.join(TARGET_PROTEIN, "csvs", f"{TARGET_PROTEIN}_scaffolds_sorted.csv")
    save_enriched_csv(scaffolds_sorted, sorted_csv_path)
    print(f"  ✓ Sorted scaffolds saved → {sorted_csv_path}")

    save_scaffold_summary_txt(scaffolds_sorted, TARGET_PROTEIN, TARGET_PROTEIN)

    # Open selector to choose scaffolds
    print("\nOpening scaffold selector...")
    # selected_scaffolds = show_scaffold_selector(TARGET_PROTEIN, scaffolds_sorted)
    selected_scaffolds = show_scaffold_selector(TARGET_PROTEIN, scaffolds_sorted, img_size=180, cols=5)

    if selected_scaffolds:
        # Save selected scaffolds
        selected_csv_path = os.path.join(TARGET_PROTEIN, "csvs", f"{TARGET_PROTEIN}_scaffolds_selected.csv")
        save_enriched_csv(selected_scaffolds, selected_csv_path)
        print(f"  ✓ Selected scaffolds saved → {selected_csv_path}")
        print(f"  → {len(selected_scaffolds)} scaffolds selected")
    else:
        print("  ⚠ No scaffolds selected")