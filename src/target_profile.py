import requests
import json

OLLAMA_URL   = "http://localhost:11434/api/generate"   # adjust if needed
QWEN_MODEL   = "qwen2.5:7b-instruct"

# ── 1. UniProt ────────────────────────────────────────────────────────────────

def fetch_uniprot(protein_name: str) -> dict:
    search_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{protein_name} AND organism_id:9606 AND reviewed:true",
        #          ^^^^^^^^^^^ scope the search to gene names only
        "fields": "accession,protein_name,gene_names,cc_function,cc_disease,cc_pathway,cc_tissue_specificity,cc_subunit",
        "format": "json",
        "size": 1,
    }
    # ... rest unchanged
    resp = requests.get(search_url, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return {}

    entry = results[0]
    accession = entry.get("primaryAccession", "")
    print(accession)
    comments = entry.get("comments", [])
    info = {"accession": accession, "comments": {}}

    for c in comments:
        ctype = c.get("commentType", "")
        if ctype == "FUNCTION":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            info["comments"]["function"] = " ".join(texts)
        elif ctype == "DISEASE":
            disease = c.get("disease", {})
            info["comments"].setdefault("diseases", []).append({
                "name":        disease.get("diseaseId", ""),
                "description": disease.get("description", ""),
            })
        elif ctype == "PATHWAY":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            info["comments"].setdefault("pathways", []).extend(texts)
        elif ctype == "TISSUE SPECIFICITY":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            info["comments"]["tissue_specificity"] = " ".join(texts)
        elif ctype == "SUBUNIT":
            texts = [t.get("value", "") for t in c.get("texts", [])]
            info["comments"]["subunit"] = " ".join(texts)

    print(info)
    return info
# ── 2. Open Targets ───────────────────────────────────────────────────────────

def fetch_open_targets(protein_name: str) -> dict:
    url = "https://api.platform.opentargets.org/api/v4/graphql"

    # Step 1: search by gene symbol to get Ensembl ID
    search_query = """
    query Search($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
        hits {
          id
          name
        }
      }
    }
    """
    resp = requests.post(url, json={"query": search_query, "variables": {"q": protein_name}}, timeout=30)
    resp.raise_for_status()
    hits = resp.json().get("data", {}).get("search", {}).get("hits", [])
    if not hits:
        print(f"    [Open Targets] No results found for {protein_name}")
        return {}

    target_id = hits[0]["id"]   # Ensembl ID e.g. ENSG00000136783

    # Step 2: fetch target details with correct current schema
    detail_query = """
    query TargetDetail($id: String!) {
      target(ensemblId: $id) {
        id
        approvedName
        approvedSymbol
        biotype
        associatedDiseases(page: {index: 0, size: 10}) {
          rows {
            disease {
              id
              name
              therapeuticAreas {
                name
              }
            }
            score
          }
        }
        knownDrugs(size: 10) {
          rows {
            drug {
              id
              name
              maximumClinicalTrialPhase
              adverseEvents {
                rows {
                  name
                  score
                }
              }
            }
            disease {
              name
            }
            mechanismOfAction
          }
        }
        tractability {
          label
          modality
          value
        }
      }
    }
    """
    resp = requests.post(url,
                         json={"query": detail_query, "variables": {"id": target_id}},
                         timeout=30)

    # Surface GraphQL-level errors before raise_for_status
    data = resp.json()
    if "errors" in data:
        print(f"    [Open Targets] GraphQL errors: {data['errors']}")
        return {}
    resp.raise_for_status()

    target_data = data.get("data", {}).get("target", {})
    if not target_data:
        return {}

    diseases = [
        {
            "name":  row["disease"]["name"],
            "score": round(row["score"], 3),
            "areas": [a["name"] for a in row["disease"].get("therapeuticAreas", [])],
        }
        for row in target_data.get("associatedDiseases", {}).get("rows", [])
    ]

    drugs = [
        {
            "name":    row["drug"]["name"],
            "phase":   row["drug"].get("maximumClinicalTrialPhase"),
            "disease": row["disease"]["name"],
            "moa":     row.get("mechanismOfAction", ""),
            "side_effects": [
                ae["name"] for ae in 
                (row["drug"].get("adverseEvents") or {}).get("rows", [])[:5]
            ],
        }
        for row in target_data.get("knownDrugs", {}).get("rows", [])
    ]

    tractability = [
        t for t in target_data.get("tractability", []) if t.get("value")
    ]

    return {
        "target_id":   target_id,
        "symbol":      target_data.get("approvedSymbol", protein_name),
        "biotype":     target_data.get("biotype", ""),
        "diseases":    diseases,
        "drugs":       drugs,
        "tractability": tractability,
    }

# ── 3. KEGG ───────────────────────────────────────────────────────────────────

def fetch_kegg_pathways(gene_symbol: str) -> list[str]:
    """Return KEGG pathway names associated with the gene."""
    # Find KEGG gene ID
    find_url = f"https://rest.kegg.jp/find/hsa/{gene_symbol}"
    resp = requests.get(find_url, timeout=30)
    if not resp.ok or not resp.text.strip():
        return []

    # Take first hit
    first_line = resp.text.strip().split("\n")[0]
    kegg_id = first_line.split("\t")[0]  # e.g. hsa:XXXX

    # Get pathways for that gene
    link_url = f"https://rest.kegg.jp/link/pathway/{kegg_id}"
    resp = requests.get(link_url, timeout=30)
    if not resp.ok or not resp.text.strip():
        return []

    pathway_ids = [line.split("\t")[1] for line in resp.text.strip().split("\n") if "\t" in line]

    # Resolve pathway names
    pathways = []
    for pid in pathway_ids[:10]:  # cap at 10
        info_url = f"https://rest.kegg.jp/get/{pid}"
        r = requests.get(info_url, timeout=20)
        if r.ok:
            for line in r.text.split("\n"):
                if line.startswith("NAME"):
                    pathways.append(line.replace("NAME", "").strip().rstrip(" - Homo sapiens (human)"))
                    break

    return pathways


# ── 4. Synthesise with Claude ─────────────────────────────────────────────────

def synthesise_target_profile(
    protein_name: str,
    uniprot: dict,
    open_targets: dict,
    kegg_pathways: list[str],
) -> str:

    raw_data = {
        "protein": protein_name,
        "uniprot": uniprot,
        "open_targets": {
            **{k: v for k, v in open_targets.items() if k != "drugs"},
            "drug_landscape": [
                {k: v for k, v in d.items() if k != "name"}  # drop drug name, keep moa/phase/side_effects
                for d in open_targets.get("drugs", [])
            ]
        },
        "kegg_pathways": kegg_pathways,
    }

    prompt = f"""You are a drug discovery expert. Based on the structured data below 
collected from UniProt, Open Targets, and KEGG, write a factual target profile 
for {protein_name}.

Present the information in complete sentences under these sections, stating only what 
the data shows. Do not add conclusions, summaries, gaps analysis, or any evaluative 
commentary. Just present the facts.

Only include a section if the data contains relevant information for it. 
If a section has no data to report, omit it entirely. Do not write placeholder 
text like "no data available" or "information is lacking".

Sections:
1. **Protein Overview** — what it is, gene name, biotype, where it's expressed
2. **Molecular Function & Mechanism** — what the protein does biochemically
3. **Disease Associations** — which diseases it's implicated in (use Open Targets scores to indicate strength of evidence)
4. **Signalling Pathways** — which pathways it participates in and its role
5. **Drug Discovery Landscape** — for each drug candidate in the data, state its mechanism of action, clinical stage, and any associated side effects. Do not name any specific drug candidates, refer to them as "a candidate", "one agent", "another compound" etc.
6. **Therapeutic Rationale** — why this is a compelling or challenging drug target

For side effects, list only what appears in the data, do not infer or add from general knowledge.

RAW DATA:
{json.dumps(raw_data, indent=2)}
"""

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":  QWEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,  # longer than filtering — needs full narrative
            },
        },
        timeout=300,  # give it time, this is a longer generation
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

