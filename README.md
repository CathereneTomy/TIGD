# TIGD
This is a computational drug discovery intelligence platform that takes a protein target name as input and automatically generates a comprehensive research report covering everything from clinical landscape to novel molecule generation.

## What It Does (End-to-End)

### 1. Target Profile

Pulls structured biological information about the target protein from:

- UniProt
- Open Targets
- KEGG

Extracted information includes:

- Gene function
- Molecular mechanism
- Disease associations
- Therapeutic rationale

An LLM formats this information into a readable narrative organized into headed sections.

---

### 2. Clinical Trials Landscape

Queries the ClinicalTrials.gov API for every study associated with the target.

Processes and categorizes studies by:

- Status
- Intervention type
- Phase
- Sponsor
- Condition

Produces:

- Trial trend chart over time
- Top associated diseases
- Small molecules in trials (active within the last 5 years)
- Intervention analysis
- Combination therapies/products
- Lead sponsor activity charts
- Collaborating institution charts
- Withdrawn studies with reasons

Also downloads:

- Clinical trial data (CSV)
- Patent data (CSV, via EPO Espacenet)

---

### 3. Drug Discovery Landscape

Runs a literature analysis through PubMed to identify which drug discovery dimensions have been explored for the target during the last 5 years.

Example dimensions include:

- Allosteric modulation
- PROTACs
- Covalent inhibition
- CNS penetration
- Selectivity

For each dimension, the system:

- Scores exploration volume
- Scores recency
- Highlights key trends
- Surfaces must-read cross-cutting papers

Outputs include clickable PubMed links.

---

### 4. ChEMBL Data Processing

#### 4.1 Clinical & Approved Molecules

Fetches all Phase 1+ molecules and:

- Maps them to PDB crystal structures (human and mouse)
- Identifies off-target proteins co-crystallized with these molecules
- Highlights potential selectivity liabilities

#### 4.2 Activity Data Processing

Processes raw assay data across standard assay types:

- IC50
- Ki
- Kd
- EC50
- Inhibition
- Others

Pipeline steps:

1. Apply potency thresholds to remove weak or inactive measurements.
2. Select the most potent molecule per document.
3. Compute Bemis–Murcko scaffolds per document.
4. Generate one CSV per assay type.
5. Plot publication volume by year.

---

### 5. Scaffold Mining

Searches PubMed for the target and:

1. Screens paper titles using a local LLM.
2. Identifies medicinal chemistry-relevant publications.
3. Extracts scaffold names from selected papers.
4. Fetches canonical SMILES from:
   - PubChem
   - CACTUS
5. Deduplicates structures.
6. Produces a sorted scaffold library.

---

### 6. Scaffold Clustering

Computes Morgan fingerprints for all mined scaffolds and performs clustering using Butina clustering with Tanimoto similarity.

Performs dimensionality reduction using one or more of:

- UMAP
- MDS
- t-SNE
- PCA

Outputs:

- Chemical space scatter plot
- Convex hulls around clusters
- Molecule grid of cluster representatives

Also identifies the central scaffold:

- The medoid of the cluster exhibiting the highest unique scaffold diversity

---

### 7. Scaffold Hopping

Runs a pharmacophore similarity pipeline that:

1. Aligns the reference molecule to the scaffold.
2. Transfers functional groups from the reference molecule onto the scaffold framework.
3. Generates a "hopped" molecule.

The resulting molecule is a new chemical entity that combines:

- The scaffold's core structure
- The reference molecule's pharmacophoric features

---

### 8. Fragment Replacement (FARE)

For each hopped molecule, the system:

1. Identifies attachment points.
2. Replaces substituents using fragments from a curated fragment library.
3. Filters fragments based on physicochemical similarity to the original substituents.
4. Generates combinatorial analogue libraries.

Outputs are saved as CSV files.

---

### 9. PDF Report Assembly

All generated outputs are consolidated into a structured JSON representation and assembled into a polished, navigable PDF report.

The final report integrates:

- Target biology
- Clinical landscape
- Drug discovery landscape
- ChEMBL analysis
- Scaffold mining
- Scaffold clustering
- Scaffold hopping
- Fragment replacement results
- Supporting visualizations and datasets
