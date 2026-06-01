"""
Pharmacophore Similarity Pipeline
Converted from pharmacophore_pipeline_stepwise.ipynb

This module contains all functions needed for pharmacophore-based molecular
similarity computation.
"""

import os
import warnings
import numpy as np
from collections import defaultdict
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, rdMolDescriptors,
    AllChem, BRICS, ChemicalFeatures
)
import pandas as pd
import re

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx

warnings.filterwarnings("ignore")


# =============================================================================
# Constants
# =============================================================================
TYPE_PENALTY = 1e6
DEFAULT_SIGMA = 1.5
DEFAULT_K = 3
DEFAULT_DIST_BINS = 10
DEFAULT_MAX_DIST = 10.0
DEFAULT_COST_THRESHOLD = 4.0
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_COORD_TOL = 0.01

FAMILY_MAP = {
    "Donor":            "HBD",
    "Acceptor":         "HBA",
    "Aromatic":         "AR",       # ← map the family directly
    "Hydrophobe":       "HP",
    "LumpedHydrophobe": "HP",
    "PosIonizable":     "PI",
    "NegIonizable":     "NI",
    "RingShape":        "RingShape",
}

FP_FTYPES = sorted(set(FAMILY_MAP.values()))

FTYPE_TO_IDX = {
    ftype: i
    for i, ftype in enumerate(FP_FTYPES)
}

FTYPE_WEIGHT = 4.0

fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


# =============================================================================
# Pharmacophore Feature Extraction
# =============================================================================
def _get_ring_shape_features(mol, conf):
    ring_shape_feats = []
    for atom_ring in mol.GetRingInfo().AtomRings():
        coords   = np.array([list(conf.GetAtomPosition(i)) for i in atom_ring])
        centroid = coords.mean(axis=0)[:2]
        ring_shape_feats.append({
            "type":         "RingShape",   # ← no size suffix
            "family":       "RingShape",
            "coords":       centroid,
            "atom_indices": sorted(atom_ring),
            "atom_set":     frozenset(atom_ring),
        })
    return ring_shape_feats


def extract_pharmacophore_features(mol, factory, remove_duplicates=True,
                                   coord_tol=DEFAULT_COORD_TOL):
    """
    Extract pharmacophore features from a molecule.

    Each feature carries an ``all_types`` set — the union of all pharmacophore
    families that share the same atom set — which is later stored in the graph
    node and used for type-aware signature computation.
    
    OPTIMIZED: Use spatial hashing for faster duplicate detection.
    """
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    # ---- collect raw features ------------------------------------------------
    raw = []
    for feat in factory.GetFeaturesForMol(mol):
        family   = feat.GetFamily()
        # raw_type = feat.GetType()                         # e.g. "Arom5", "Arom6", "SingleAtomDonor"

        # For aromatics, use the size-specific type directly.
        # For everything else, map via FAMILY_MAP as before.
        if family == "Aromatic":
            ftype = "AR"   # ← was FAMILY_MAP.get(raw_type, raw_type) which gave AR5/AR6 etc.
        else:
            ftype = FAMILY_MAP.get(family, family)
        atom_ids = frozenset(feat.GetAtomIds())
        coords  = np.array([list(conf.GetAtomPosition(i)) for i in atom_ids])
        centroid = coords.mean(axis=0)[:2]
        raw.append({
            "type":         ftype,
            "family":       family,
            "coords":       centroid,
            "atom_indices": sorted(atom_ids),
            "atom_set":     atom_ids,
        })
    raw += _get_ring_shape_features(mol, conf)

    # ---- annotate all_types per atom-set -------------------------------------
    atom_set_groups = defaultdict(list)
    for feat in raw:
        atom_set_groups[feat["atom_set"]].append(feat)

    for group in atom_set_groups.values():
        all_types = frozenset(FAMILY_MAP.get(f["family"], f["family"]) for f in group)
        for feat in group:
            feat["all_types"] = all_types

    # ---- coordinate-based duplicate removal (OPTIMIZED) -----
    features = []
    if remove_duplicates:
        # Use spatial hashing grid to avoid O(n²) pairwise comparisons
        # Build a grid with cell size = coord_tol
        coord_grid = {}  # (grid_x, grid_y, ftype) -> feature
        
        for feat in raw:
            ftype = feat["type"]
            x, y = feat["coords"]
            # Snap coordinates to grid
            grid_x = int(np.round(x / coord_tol))
            grid_y = int(np.round(y / coord_tol))
            
            # Check nearby grid cells for duplicates (3x3 neighborhood)
            is_duplicate = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (grid_x + dx, grid_y + dy, ftype)
                    if cell in coord_grid:
                        existing = coord_grid[cell]
                        if np.linalg.norm(existing["coords"] - feat["coords"]) < coord_tol:
                            is_duplicate = True
                            break
                if is_duplicate:
                    break
            
            if not is_duplicate:
                coord_grid[(grid_x, grid_y, ftype)] = feat
                features.append(feat)
    else:
        features = raw

    # drop internal key not needed downstream
    for f in features:
        f.pop("atom_set", None)

    return features

# =============================================================================
# Pharmacophore Graph
# =============================================================================

def build_pharmacophore_graph(features):
    """
    Build a fully-connected graph from pharmacophore features.
    Node attributes include ``all_types`` so that signature computation
    can correctly reflect multi-family atom sets.
    
    OPTIMIZED: Vectorized distance computation for edges.
    """
    G = nx.Graph()
    for i, feat in enumerate(features):
        G.add_node(
            i,
            ftype=feat["type"],
            family=feat["family"],
            coords=feat["coords"],
            atom_indices=feat["atom_indices"],
            all_types=feat.get("all_types", frozenset({feat["type"]})),
        )

    # VECTORIZED: Compute all pairwise distances at once
    coords = np.array([f["coords"] for f in features])
    if len(features) > 1:
        # Compute pairwise distances: (n, n)
        pairwise_dists = cdist(coords, coords, metric="euclidean")
        
        # Add edges for upper triangle only (fully connected undirected graph)
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dist = float(pairwise_dists[i, j])
                G.add_edge(i, j, distance=dist)
    return G


# =============================================================================
# Modifying the graph to see overlapped nodes
# =============================================================================

def tag_overlapping_nodes(graph):
    """
    For each pair of nodes whose atom_indices intersect in a true embedding
    (one is a proper subset of the other, or more than 2 atoms shared),
    tag ONLY the smaller node (the embedded feature) with:

        is_overlapped      : True
        overlapping_ftypes : set of ftypes of the larger nodes it sits inside

    The larger/container node is left untagged.
    Fused ring bonds (exactly 1 or 2 shared atoms, neither a subset) are ignored.

    Modifies graph in-place.
    
    OPTIMIZED: Pre-convert atom_indices to frozensets during initialization,
    cache node atom sets.
    """
    nodes = list(graph.nodes())

    for n in nodes:
        graph.nodes[n]["is_overlapped"]      = False
        graph.nodes[n]["overlapping_ftypes"] = set()

    # OPTIMIZATION: Pre-cache atom sets as frozensets for faster comparison
    atom_sets = {
        n: frozenset(graph.nodes[n].get("atom_indices", []))
        for n in nodes
    }

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj   = nodes[i], nodes[j]
            atoms_i  = atom_sets[ni]
            atoms_j  = atom_sets[nj]

            shared = atoms_i & atoms_j
            if not shared:
                continue

            is_subset    = atoms_i < atoms_j or atoms_j < atoms_i
            deep_overlap = len(shared) > 2

            if not (is_subset or deep_overlap):
                continue   # fusion bond — skip

            # Tag only the larger node
            if len(atoms_i) >= len(atoms_j):
                larger, smaller = ni, nj
            else:
                larger, smaller = nj, ni

            graph.nodes[larger]["is_overlapped"] = True
            graph.nodes[larger]["overlapping_ftypes"].add(
                graph.nodes[smaller]["ftype"]
            )

def tag_fused_nodes(mol, graph):
    """
    Tag each graph node with is_fused (bool) indicating whether it is
    part of a fused ring system.
 
    Rules
    -----
    Ring node (atom_indices > 1):
        is_fused = True if its atom set shares >= 2 atoms with ANY
        other ring in the molecule (i.e. it shares a fusion bond).
 
    Single-atom node (atom_indices == 1):
        is_fused = True if that atom sits at the junction of two rings
        — i.e. it appears in >= 2 distinct rings in the molecule.
        These are the bridgehead/fusion atoms (e.g. the shared carbons
        in benzimidazole, or the nitrogen at the junction of a
        pyrimidine fused to a pyridone).
 
    Modifies graph in-place.
    """
    ring_info = mol.GetRingInfo()
    all_rings = [set(r) for r in ring_info.AtomRings()]
 
    # Precompute: how many rings each atom belongs to
    atom_ring_count = {}
    for ring in all_rings:
        for atom_idx in ring:
            atom_ring_count[atom_idx] = atom_ring_count.get(atom_idx, 0) + 1
 
    for n in graph.nodes():
        atom_indices = graph.nodes[n].get("atom_indices", [])
        atom_set     = set(atom_indices)
        is_fused     = False
 
        if len(atom_indices) > 1:
            # Ring node — check if it shares a fusion bond (>=2 atoms)
            # with any other ring
            for ring in all_rings:
                if ring == atom_set:
                    continue   # same ring, skip
                shared = atom_set & ring
                if len(shared) >= 2:
                    is_fused = True
                    break
 
        elif len(atom_indices) == 1:
            # Single-atom node — fused if the atom sits in >= 2 rings
            atom_idx = list(atom_indices)[0]
            if atom_ring_count.get(atom_idx, 0) >= 2:
                is_fused = True
 
        graph.nodes[n]["is_fused"] = is_fused

def effective_ftype(graph, node):
    """
    Return the effective type string for penalty computation.

    Format:
        standalone         → "HBA"
        overlapped         → "HBA|AR6"  (sorted partners joined by |)
                          or "HBA|AR6|RingShape6" if multiple overlaps

    The base ftype always comes first. Partners are sorted for
    consistent comparison.
    """
    data = graph.nodes[node]
    base = data["ftype"]
    partners = data.get("overlapping_ftypes", set())

    if not partners:
        return base

    sorted_partners = "|".join(sorted(partners))
    return f"{base}|{sorted_partners}"

# =============================================================================
# Extracting all the unique ftypes in the given graphs
# =============================================================================

def collect_unique_types(*graphs):
    """Return the sorted union of all feature types across the supplied graphs."""
    all_type_set = set()
    for G in graphs:
        for n in G.nodes():
            all_type_set.update(G.nodes[n].get("all_types", {G.nodes[n]["ftype"]}))
    return sorted(all_type_set)

# =============================================================================
# Extracting signatures for each node to compute the cost
# =============================================================================

def compute_node_signatures(graph, k=DEFAULT_K, dist_bins=DEFAULT_DIST_BINS,
                             max_dist=DEFAULT_MAX_DIST, unique_types=None):
    """
    Compute a fixed-length descriptor for every node:
      [k-NN distances | distance histogram | neighbour type histogram]

    ``unique_types`` should span *both* the reference and query graphs so that
    signatures are comparable across molecules.
    
    OPTIMIZED: More efficient type histogram computation using list accumulation.
    """
    nodes = list(graph.nodes())

    if unique_types is None:
        unique_types = collect_unique_types(graph)

    n_types = len(unique_types)
    type_index = {t: idx for idx, t in enumerate(unique_types)}

    signatures = {}
    for node in nodes:
        neighbour_dists = sorted(graph[node][nb]["distance"] for nb in graph.neighbors(node))
        knn_dists = np.array(
            neighbour_dists[:k] + [0.0] * max(0, k - len(neighbour_dists))
        )
        dist_hist, _ = np.histogram(neighbour_dists, bins=dist_bins,
                                    range=(0, max_dist), density=True)

        # OPTIMIZED: Accumulate type indices first, then build histogram
        type_indices = []
        for nb in graph.neighbors(node):
            for t in graph.nodes[nb].get("all_types", {graph.nodes[nb]["ftype"]}):
                if t in type_index:
                    type_indices.append(type_index[t])
        
        # Use np.bincount for efficient histogram computation
        if type_indices:
            type_hist = np.bincount(type_indices, minlength=n_types)
        else:
            type_hist = np.zeros(n_types)

        sig = np.concatenate([
            knn_dists,
            dist_hist / (dist_hist.sum() + 1e-9),
            type_hist / (type_hist.sum() + 1e-9),
        ])
        signatures[node] = sig

    return signatures, unique_types

# Penalty constants — OVERLAP_CONTEXT_PENALTY sits between 0 and SOFT_TYPE_PENALTY
OVERLAP_CONTEXT_PENALTY = 0.4 * TYPE_PENALTY

# =============================================================================
# Penalty for non-overlapping features when paired with overlapping ones
# =============================================================================

AR_SIZES = {"AR4": 4, "AR5": 5, "AR6": 6, "AR7": 7, "AR8": 8}

def soft_type_penalty(graph_a, node_a, graph_b, node_b):
    eff_a = effective_ftype(graph_a, node_a)
    eff_b = effective_ftype(graph_b, node_b)

    # --- Layer 1: exact match (base + same partners) ---
    if eff_a == eff_b:
        return 0.0

    # --- Extract base type and partner sets ---
    def parse_effective(eff):
        parts = eff.split("|")
        return parts[0], set(parts[1:])   # base, partners

    base_a, partners_a = parse_effective(eff_a)
    base_b, partners_b = parse_effective(eff_b)

    ol_a = bool(partners_a)
    ol_b = bool(partners_b)

    # --- Layer 2: cluster check on base types ---
    cluster_a = FTYPE_TO_CLUSTER.get(base_a, -1)
    cluster_b = FTYPE_TO_CLUSTER.get(base_b, -2)

    if cluster_a != cluster_b:
        return HARD_TYPE_PENALTY

    # --- Same cluster: base penalty ---
    if base_a == base_b:
        base_penalty = 0.0
    elif base_a in AR_SIZES and base_b in AR_SIZES:
        size_diff = abs(AR_SIZES[base_a] - AR_SIZES[base_b])
        base_penalty = (size_diff / 4.0) * SOFT_TYPE_PENALTY
    else:
        base_penalty = SOFT_TYPE_PENALTY

    # --- Layer 3: overlap context comparison ---
    if not ol_a and not ol_b:
        # both standalone — no overlap penalty
        overlap_penalty = 0.0

    elif ol_a != ol_b:
        # one is overlapped, the other is standalone
        overlap_penalty = OVERLAP_CONTEXT_PENALTY

    else:
        # both overlapped — check if their partner clusters agree
        # get the cluster set for each node's partners
        partner_clusters_a = {FTYPE_TO_CLUSTER.get(p, -1) for p in partners_a}
        partner_clusters_b = {FTYPE_TO_CLUSTER.get(p, -2) for p in partners_b}

        if partner_clusters_a == partner_clusters_b:
            # partners are in the same clusters — good match, small penalty
            overlap_penalty = 0.0
        elif partner_clusters_a & partner_clusters_b:
            # partial overlap in partner clusters — partial penalty
            overlap_penalty = 0.5 * OVERLAP_CONTEXT_PENALTY
        else:
            # partners are in completely different clusters
            # e.g. HBA overlapping AR6 vs HBA overlapping RingShape3
            overlap_penalty = OVERLAP_CONTEXT_PENALTY

    return min(base_penalty + overlap_penalty, HARD_TYPE_PENALTY)

from rdkit.Chem import Crippen, rdPartialCharges

# def node_fingerprint(mol, atom_indices):
#     # Get Crippen contribs once for whole mol
#     crippen_contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    
#     # Gasteiger charges
#     rdPartialCharges.ComputeGasteigerCharges(mol)
    
#     atoms = [mol.GetAtomWithIdx(i) for i in atom_indices]
#     n = len(atoms)
    
#     # Per-atom features, aggregated over the feature's atom set
#     atomic_nums    = [a.GetAtomicNum() for a in atoms]
#     charges        = [float(a.GetPropsAsDict().get('_GasteigerCharge', 0)) for a in atoms]
#     formal_charges = [a.GetFormalCharge() for a in atoms]
#     is_aromatic    = [int(a.GetIsAromatic()) for a in atoms]
#     hybridization  = [int(a.GetHybridization()) for a in atoms]
#     degree         = [a.GetDegree() for a in atoms]
#     logp_contribs  = [crippen_contribs[i][0] for i in atom_indices]
#     mr_contribs    = [crippen_contribs[i][1] for i in atom_indices]
#     is_hbd = [int(a.GetTotalNumHs() > 0 and a.GetAtomicNum() in (7, 8)) for a in atoms]
#     is_hba = [int(a.GetAtomicNum() in (7, 8) and a.GetTotalNumHs() == 0) for a in atoms]

#     return np.array([
#         1.5* np.mean(atomic_nums),
#         np.std(atomic_nums),
#         np.mean(charges),
#         np.sum(formal_charges),
#         2.0 * np.mean(is_aromatic),
#         np.mean(hybridization),
#         1.5* n,
#         np.mean(degree),
#         np.sum(logp_contribs),
#         np.sum(mr_contribs),
#         np.mean(is_hbd),    # ← new
#         np.mean(is_hba),    # ← new
#     ])

# def node_fingerprint(mol, node):
#     """
#     Chemistry-aware fingerprint for a pharmacophore node.

#     Returns
#     -------
#     np.ndarray
#     """

#     atom_indices = node["atom_indices"]

#     atoms = [
#         mol.GetAtomWithIdx(i)
#         for i in atom_indices
#     ]

#     # -------------------------------------------------------------------------
#     # Basic atomic properties
#     # -------------------------------------------------------------------------

#     atomic_nums = np.array(
#         [a.GetAtomicNum() for a in atoms],
#         dtype=float,
#     )

#     formal_charges = np.array(
#         [a.GetFormalCharge() for a in atoms],
#         dtype=float,
#     )

#     aromatic_flags = np.array(
#         [float(a.GetIsAromatic()) for a in atoms],
#         dtype=float,
#     )

#     hybridizations = np.array(
#         [float(int(a.GetHybridization())) for a in atoms],
#         dtype=float,
#     )

#     degrees = np.array(
#         [a.GetDegree() for a in atoms],
#         dtype=float,
#     )

#     # -------------------------------------------------------------------------
#     # Gasteiger charges
#     # -------------------------------------------------------------------------

#     try:
#         rdPartialCharges.ComputeGasteigerCharges(mol)

#         gasteiger = np.array(
#             [
#                 float(a.GetProp("_GasteigerCharge"))
#                 if a.HasProp("_GasteigerCharge")
#                 else 0.0
#                 for a in atoms
#             ],
#             dtype=float,
#         )

#         gasteiger = np.nan_to_num(gasteiger)

#     except Exception:
#         gasteiger = np.zeros(len(atoms))

#     # -------------------------------------------------------------------------
#     # Crippen atom contributions
#     # -------------------------------------------------------------------------

#     try:
#         contribs = rdMolDescriptors._CalcCrippenContribs(mol)

#         logp_contribs = np.array(
#             [contribs[i][0] for i in atom_indices],
#             dtype=float,
#         )

#         mr_contribs = np.array(
#             [contribs[i][1] for i in atom_indices],
#             dtype=float,
#         )

#     except Exception:

#         logp_contribs = np.zeros(len(atoms))
#         mr_contribs = np.zeros(len(atoms))

#     # -------------------------------------------------------------------------
#     # Chemistry descriptor block
#     # -------------------------------------------------------------------------

#     chem_fp = np.array(
#         [
#             # atomic composition
#             1.5 * np.mean(atomic_nums),
#             np.std(atomic_nums),

#             # electrostatics
#             np.mean(gasteiger),
#             np.sum(formal_charges),

#             # aromatic / hybridization character
#             2.0 * np.mean(aromatic_flags),
#             np.mean(hybridizations),

#             # feature size
#             1.5 * len(atom_indices),

#             # local connectivity
#             np.mean(degrees),

#             # lipophilicity / polarizability
#             np.sum(logp_contribs),
#             np.sum(mr_contribs),
#         ],
#         dtype=float,
#     )

#     # -------------------------------------------------------------------------
#     # Pharmacophore-type one-hot block
#     # -------------------------------------------------------------------------

#     ftype_vec = np.zeros(
#         len(FP_FTYPES),
#         dtype=float,
#     )

#     ftype = node.get("ftype")

#     if ftype in FTYPE_TO_IDX:
#         ftype_vec[FTYPE_TO_IDX[ftype]] = FTYPE_WEIGHT

#     # -------------------------------------------------------------------------
#     # Final fingerprint
#     # -------------------------------------------------------------------------

#     fp = np.concatenate(
#         [
#             chem_fp,
#             ftype_vec,
#         ]
#     )

#     return fp  
 

FUSED_WEIGHT = 2.0
 
def node_fingerprint(mol, node):
    """
    Chemistry-aware fingerprint for a pharmacophore node.
    Now includes is_fused flag (weight=2) from tag_fused_nodes.
 
    Returns
    -------
    np.ndarray
    """
    atom_indices = node["atom_indices"]
    atoms = [mol.GetAtomWithIdx(i) for i in atom_indices]
 
    # -------------------------------------------------------------------------
    # Basic atomic properties
    # -------------------------------------------------------------------------
    atomic_nums = np.array(
        [a.GetAtomicNum() for a in atoms], dtype=float)
 
    formal_charges = np.array(
        [a.GetFormalCharge() for a in atoms], dtype=float)
 
    aromatic_flags = np.array(
        [float(a.GetIsAromatic()) for a in atoms], dtype=float)
 
    hybridizations = np.array(
        [float(int(a.GetHybridization())) for a in atoms], dtype=float)
 
    degrees = np.array(
        [a.GetDegree() for a in atoms], dtype=float)
 
    # -------------------------------------------------------------------------
    # Gasteiger charges
    # -------------------------------------------------------------------------
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        gasteiger = np.array(
            [
                float(a.GetProp("_GasteigerCharge"))
                if a.HasProp("_GasteigerCharge") else 0.0
                for a in atoms
            ], dtype=float)
        gasteiger = np.nan_to_num(gasteiger)
    except Exception:
        gasteiger = np.zeros(len(atoms))
 
    # -------------------------------------------------------------------------
    # Crippen atom contributions
    # -------------------------------------------------------------------------
    try:
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        logp_contribs = np.array(
            [contribs[i][0] for i in atom_indices], dtype=float)
        mr_contribs = np.array(
            [contribs[i][1] for i in atom_indices], dtype=float)
    except Exception:
        logp_contribs = np.zeros(len(atoms))
        mr_contribs   = np.zeros(len(atoms))
 
    # -------------------------------------------------------------------------
    # Chemistry descriptor block
    # -------------------------------------------------------------------------
    chem_fp = np.array([
        1.5 * np.mean(atomic_nums),
        np.std(atomic_nums),
        np.mean(gasteiger),
        np.sum(formal_charges),
        2.0 * np.mean(aromatic_flags),
        np.mean(hybridizations),
        1.5 * len(atom_indices),
        np.mean(degrees),
        np.sum(logp_contribs),
        np.sum(mr_contribs),
    ], dtype=float)
 
    # -------------------------------------------------------------------------
    # Pharmacophore-type one-hot block  (weight = FTYPE_WEIGHT, e.g. 4)
    # -------------------------------------------------------------------------
    ftype_vec = np.zeros(len(FP_FTYPES), dtype=float)
    ftype = node.get("ftype")
    if ftype in FTYPE_TO_IDX:
        ftype_vec[FTYPE_TO_IDX[ftype]] = FTYPE_WEIGHT
 
    # -------------------------------------------------------------------------
    # Fused-ring flag  (weight = FUSED_WEIGHT = 2)
    #
    # Read from node["is_fused"] which is set by tag_fused_nodes().
    # A fused node gets FUSED_WEIGHT in its own dedicated dimension so
    # that fused-ring features are penalised when matched against
    # non-fused features of the same type.
    # -------------------------------------------------------------------------
    is_fused     = float(node.get("is_fused", False))
    fused_vec    = np.array([FUSED_WEIGHT * is_fused], dtype=float)
 
    # -------------------------------------------------------------------------
    # Final fingerprint
    # -------------------------------------------------------------------------
    fp = np.concatenate([chem_fp, ftype_vec, fused_vec])
    return fp

# def compute_fingerprint_type_cost(ref_nodes, query_nodes, ref_graph, query_graph, ref_mol, query_mol):
    
#     # Precompute fingerprints for all nodes
#     ref_fps = {
#         n: node_fingerprint(ref_mol, ref_graph.nodes[n]['atom_indices'])
#         for n in ref_nodes
#     }
#     query_fps = {
#         n: node_fingerprint(query_mol, query_graph.nodes[n]['atom_indices'])
#         for n in query_nodes
#     }
    
#     ref_mat   = np.array([ref_fps[n]   for n in ref_nodes])
#     query_mat = np.array([query_fps[n] for n in query_nodes])
    
#     # Raw distance
#     raw_dist = cdist(ref_mat, query_mat, metric='euclidean')
    
#     # Scale to same range as old type penalties
#     # So it doesn't get drowned out by sig_cost
#     row_max = raw_dist.max(axis=1, keepdims=True)
#     type_cost = raw_dist / (row_max + 1e-9)
#     print(f"ref_nodes count:   {len(ref_nodes)}")
#     print(f"query_nodes count: {len(query_nodes)}")
#     print(f"raw_dist shape:    {raw_dist.shape}")
#     print(type_cost)
#     return type_cost

def compute_fingerprint_type_cost(ref_nodes, query_nodes, ref_graph, 
                                   query_graph, ref_mol, query_mol):
    
    ref_fps = {n: node_fingerprint(ref_mol, ref_graph.nodes[n]) 
               for n in ref_nodes}
    query_fps = {n: node_fingerprint(query_mol, query_graph.nodes[n]) 
                 for n in query_nodes}
    
    ref_mat   = np.array([ref_fps[n] for n in ref_nodes])
    query_mat = np.array([query_fps[n] for n in query_nodes])
    
    raw_dist = cdist(ref_mat, query_mat, metric='euclidean')
    
    # Normalize FIRST — before fused penalty
    row_max   = raw_dist.max(axis=1, keepdims=True)
    type_cost = raw_dist / (row_max + 1e-9)
    
    # Add fused mismatch penalty AFTER normalization
    # so it isn't diluted by row_max
    for i, rn in enumerate(ref_nodes):
        for j, qn in enumerate(query_nodes):
            ref_fused   = ref_graph.nodes[rn].get("is_fused", False)
            query_fused = query_graph.nodes[qn].get("is_fused", False)
            if ref_fused != query_fused:
                type_cost[i, j] += 0.5   # ← added after normalization
    
    return type_cost
# # =============================================================================
# Initial Cost Matrix with a single type penalty
# =============================================================================

def compute_cost_matrix(ref_graph, query_graph, ref_sigs, query_sigs,ref_mol, query_mol):
    """
    Compute the assignment cost matrix combining:
      - Euclidean signature distance
      - Geometric consistency (pairwise distance agreement)
      - Hard type-mismatch penalty (TYPE_PENALTY)
    
    OPTIMIZED: Vectorized consistency cost computation using numpy broadcasting
    instead of nested Python loops.
    """
    ref_nodes   = list(ref_graph.nodes())
    query_nodes = list(query_graph.nodes())

    ref_mat   = np.array([ref_sigs[n]   for n in ref_nodes])
    query_mat = np.array([query_sigs[n] for n in query_nodes])
    sig_cost  = cdist(ref_mat, query_mat, metric="euclidean")

    ref_coords   = np.array([ref_graph.nodes[n]["coords"]   for n in ref_nodes])
    query_coords = np.array([query_graph.nodes[n]["coords"] for n in query_nodes])

    D_ref   = cdist(ref_coords,   ref_coords,   metric="euclidean")
    D_query = cdist(query_coords, query_coords, metric="euclidean")

    # Geometric consistency: for each tentative (i,j) pairing, measure how well
    # the pairwise distances from i and j agree with those of the rest of the graph.
    consistency_cost = np.zeros((len(ref_nodes), len(query_nodes)))
    for i in range(len(ref_nodes)):
        for j in range(len(query_nodes)):
            best_match = np.abs(D_ref[i, :, None] - D_query[None, j, :]).min(axis=1)
            consistency_cost[i, j] = best_match.mean()

    # Precompute type cost matrix using memoized soft_type_penalty
    type_cost = compute_fingerprint_type_cost(ref_nodes, query_nodes, ref_graph, query_graph, ref_mol, query_mol)

    # type_cost = np.zeros((len(ref_nodes), len(query_nodes)))
    # for i, rn in enumerate(ref_nodes):
    #     for j, qn in enumerate(query_nodes):
    #         type_cost[i, j] = soft_type_penalty(ref_graph, rn, query_graph, qn)

    cost = sig_cost + 0.5 * consistency_cost + type_cost
    return cost, ref_nodes, query_nodes


# =============================================================================
# Initial Matching and Alignment
# =============================================================================
def hungarian_matching(cost, ref_nodes, query_nodes, ref_graph, query_graph):
    """
    Perform Hungarian matching given a pre-computed cost matrix.
    Returns deduplicated matches sorted by ascending cost.

    Accepts the already-computed (cost, ref_nodes, query_nodes) from
    ``compute_cost_matrix`` to avoid redundant recomputation.
    """
    row_ind, col_ind = linear_sum_assignment(cost)

    raw_matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < TYPE_PENALTY:
            rn, qn = ref_nodes[r], query_nodes[c]
            raw_matches.append({
                "ref_node":          rn,
                "query_node":        qn,
                "ref_type":          ref_graph.nodes[rn]["ftype"],
                "query_type":        query_graph.nodes[qn]["ftype"],
                "cost":              cost[r, c],
                "ref_atom_indices":  tuple(sorted(ref_graph.nodes[rn].get("atom_indices", []))),
                "query_atom_indices":tuple(sorted(query_graph.nodes[qn].get("atom_indices", []))),
            })
    raw_matches.sort(key=lambda x: x["cost"])

    # Deduplicate: keep the lowest-cost assignment per unique atom-index tuple
    seen_ref, seen_query = set(), set()
    matches = []
    for m in raw_matches:
        if m["ref_atom_indices"] not in seen_ref and m["query_atom_indices"] not in seen_query:
            seen_ref.add(m["ref_atom_indices"])
            seen_query.add(m["query_atom_indices"])
            matches.append(m)
    return matches

# ============================================================================================
# Aligning the molecules based on the matched pairs, calculating the ph similarity and RMSD
# ============================================================================================

def kabsch_2d(P, Q):
    """
    Optimal 2D rotation + translation that superimposes P onto Q (Kabsch algorithm).
    Returns (R, t, P_aligned) where P_aligned = P @ R.T + t.
    """
    P_c, Q_c = P.mean(axis=0), Q.mean(axis=0)
    P_, Q_ = P - P_c, Q - Q_c
    H = P_.T @ Q_
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = Q_c - P_c @ R.T
    return R, t, P @ R.T + t


def _align_and_rmsd(ref_graph, query_graph, matches):
    """
    Align matched query features onto reference features and return RMSD.
    Returns 0.0 when only one match exists (trivially perfect alignment).
    """
    if len(matches) == 1:
        return 0.0

    ref_coords   = np.array([ref_graph.nodes[m["ref_node"]]["coords"]   for m in matches])
    query_coords = np.array([query_graph.nodes[m["query_node"]]["coords"] for m in matches])
    _, _, P_aligned = kabsch_2d(query_coords, ref_coords)
    diff = P_aligned - ref_coords
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def pharmacophore_similarity_score(rmsd, sigma=DEFAULT_SIGMA):
    """Gaussian-decay similarity score: exp(-rmsd² / 2σ²)."""
    return float(np.exp(-rmsd ** 2 / (2 * sigma ** 2)))


# =============================================================================
# Computing the graphs and signatures
# =============================================================================

def _compute_graphs_and_sigs(ref_smiles, query_smiles):
    """
    Parse SMILES → extract features → build graphs → compute signatures.
    Returns (ref_mol, query_mol, ref_g, query_g, ref_sigs, query_sigs, all_types)
    or None on parse failure.
    """
    ref_mol   = Chem.MolFromSmiles(ref_smiles)
    query_mol = Chem.MolFromSmiles(query_smiles)
    if not ref_mol or not query_mol:
        return None

    ref_feats   = extract_pharmacophore_features(ref_mol,   factory)
    query_feats = extract_pharmacophore_features(query_mol, factory)

    ref_g   = build_pharmacophore_graph(ref_feats)
    query_g = build_pharmacophore_graph(query_feats)

    if ref_g.number_of_nodes() == 0 or query_g.number_of_nodes() == 0:
        return None

    all_types = collect_unique_types(ref_g, query_g)
    ref_sigs,   _ = compute_node_signatures(ref_g,   unique_types=all_types)
    query_sigs, _ = compute_node_signatures(query_g, unique_types=all_types)

    return ref_mol, query_mol, ref_g, query_g, ref_sigs, query_sigs, all_types

# =============================================================================
# Pass 1 flow for initial matches
# =============================================================================

def ph_similarity_pipeline(ref_smiles,query_smiles, ref_sigs=None, query_sigs=None,
                              sigma=DEFAULT_SIGMA, verbose=False):
    """
    Compute the pharmacophore similarity score between two pre-built graphs.

    ``ref_sigs`` / ``query_sigs`` may be supplied to avoid redundant
    signature computation.  If omitted they are computed internally using
    the union of types across both graphs.
    """

  
    result = _compute_graphs_and_sigs(ref_smiles, query_smiles)
    if result is None:
        return []
    
    ref_mol, query_mol, ref_graph, query_graph, ref_sigs, query_sigs, _ = result
    tag_overlapping_nodes(ref_graph)
    tag_overlapping_nodes(query_graph)
    tag_fused_nodes(ref_mol, ref_graph)
    tag_fused_nodes(query_mol,query_graph)


    all_types = collect_unique_types(ref_graph, query_graph)

    if ref_sigs is None:
        ref_sigs,   _ = compute_node_signatures(ref_graph,   unique_types=all_types)
    if query_sigs is None:
        query_sigs, _ = compute_node_signatures(query_graph, unique_types=all_types)

    cost, ref_nodes, query_nodes = compute_cost_matrix(ref_graph, query_graph, ref_sigs, query_sigs, ref_mol, query_mol)
    matches = hungarian_matching(cost, ref_nodes, query_nodes, ref_graph, query_graph)

    if not matches:
        return {"score": 0.0, "rmsd": np.inf, "n_matched": 0, "matches": [],
                "raw_score": 0.0, "coverage": 0.0}
    
    matched_pairs        = [
        {
            "ref_node":          m["ref_node"],
            "query_node":        m["query_node"],
            "ref_atom_indices":  ref_graph.nodes[m["ref_node"]].get("atom_indices", []),
            "query_atom_indices":query_graph.nodes[m["query_node"]].get("atom_indices", []),
            "ref_type":          m["ref_type"],
            "query_type":        m["query_type"],
            "cost":              m["cost"],
        }
        for m in matches
    ]

    rmsd       = _align_and_rmsd(ref_graph, query_graph, matches)
    raw_score  = pharmacophore_similarity_score(rmsd, sigma)
    coverage   = len(matches) / max(ref_graph.number_of_nodes(), query_graph.number_of_nodes())
    final      = raw_score * coverage

    if verbose:
        n_max = max(ref_graph.number_of_nodes(), query_graph.number_of_nodes())
        print(f"Matched: {len(matches)} / {n_max}")
        print(f"RMSD: {rmsd:.4f} Å | Raw: {raw_score:.4f} | "
              f"Coverage: {coverage:.4f} | Score: {final:.4f}")

    return {
        "score":     final,
        "raw_score": raw_score,
        "rmsd":      rmsd,
        "coverage":  coverage,
        "n_matched": len(matches),
        "matches":   matches,
        "matched_pairs": matched_pairs,
        "ref_graph": ref_graph,
        "query_graph": query_graph,
        "ref_sigs": ref_sigs,
        "query_sigs": query_sigs

    }

# =============================================================================
# Type Hierarchy and Soft Penalty
# =============================================================================

# Three-tier cluster definition.
# Types in the same cluster get a partial penalty.
# Types in different clusters get a hard penalty.

TYPE_CLUSTERS = [
    {"HBD", "HBA", "PI", "NI"},
    {"AR","RingShape"},           # ← was AR5, AR6, AR7, AR8
    {"HP"},
  # ← was RingShape3..RingShape7plus
]

FTYPE_TO_CLUSTER = {}
for cluster_idx, cluster in enumerate(TYPE_CLUSTERS):
    for ftype in cluster:
        FTYPE_TO_CLUSTER[ftype] = cluster_idx

SOFT_TYPE_PENALTY = 0.3 * TYPE_PENALTY   # same cluster, different type
HARD_TYPE_PENALTY = 1e6                  # different cluster entirely


# =============================================================================
# Spatial Consistency Filter (Pass 1 post-processing)
# =============================================================================

def spatial_consistency_filter(matches, ref_graph, query_graph, threshold=0.5):
    if not matches:
        return [], []

    anchor = matches[0]
    ref_anc_coords   = np.array(ref_graph.nodes[anchor["ref_node"]]["coords"])
    query_anc_coords = np.array(query_graph.nodes[anchor["query_node"]]["coords"])

    pass1_matches = [anchor]
    for m in matches[1:]:
        ref_coords   = np.array(ref_graph.nodes[m["ref_node"]]["coords"])
        query_coords = np.array(query_graph.nodes[m["query_node"]]["coords"])
        d_ref   = np.linalg.norm(ref_coords   - ref_anc_coords)
        d_query = np.linalg.norm(query_coords - query_anc_coords)
        if abs(d_ref - d_query) <= threshold:
            pass1_matches.append(m)

    # --- Build matched sets by BOTH node id AND atom indices ---
    matched_query_nodes      = {m["query_node"]        for m in pass1_matches}
    matched_query_atom_sets  = {m["query_atom_indices"] for m in pass1_matches}

    all_query_nodes = list(query_graph.nodes())
    unmatched_query_nodes = []
    for qn in all_query_nodes:
        if qn in matched_query_nodes:
            continue   # node id already matched
        atom_ids = tuple(sorted(query_graph.nodes[qn].get("atom_indices", [])))
        if atom_ids in matched_query_atom_sets:
            continue   # same physical atoms already matched under a different node id
        unmatched_query_nodes.append(qn)

    return pass1_matches, unmatched_query_nodes

# =============================================================================
# Spatial Window Definition
# =============================================================================

def define_spatial_window(
    pass1_matches,
    ref_graph,
    query_graph,
    buffer=0.0,
):
    """
    Define the spatial search region in ref-space for Pass 2.

    The window is centered on the ref anchor (lowest cost / first pass1 match)
    and has a radius equal to the farthest query node from the query anchor
    plus a buffer.

    Parameters
    ----------
    pass1_matches : spatially consistent matches from spatial_consistency_filter
    ref_graph     : pharmacophore graph of the reference molecule
    query_graph   : pharmacophore graph of the query molecule
    buffer        : extra Angstroms added on top of max_query_radius (default 2.0)

    Returns
    -------
    window_ref_nodes   : list of ref nodes inside the spatial window
    search_radius      : the actual radius used (max_query_radius + buffer)
    max_query_radius   : farthest distance from query anchor to any query node
    """

    # --- Anchor in both spaces (lowest cost = first in pass1_matches) ---
    anchor        = pass1_matches[0]
    ref_anc_node  = anchor["ref_node"]
    query_anc_node = anchor["query_node"]

    ref_anc_coords   = np.array(ref_graph.nodes[ref_anc_node]["coords"])
    query_anc_coords = np.array(query_graph.nodes[query_anc_node]["coords"])

    # --- Max query radius (VECTORIZED) ---
    # Compute all distances at once using numpy broadcasting
    all_query_nodes = list(query_graph.nodes())
    query_coords = np.array([query_graph.nodes[qn]["coords"] for qn in all_query_nodes])
    
    # Vectorized distance computation
    query_distances = np.linalg.norm(query_coords - query_anc_coords, axis=1)
    
    # Exclude anchor node
    mask = np.array([qn != query_anc_node for qn in all_query_nodes])
    query_distances_filtered = query_distances[mask]
    
    max_query_radius = float(np.max(query_distances_filtered)) if len(query_distances_filtered) > 0 else 0.0
    search_radius    = max_query_radius + buffer

    # --- Collect ref nodes within search_radius (VECTORIZED) ---
    # Exclude ref nodes already claimed by pass1_matches so Pass 2
    # only considers unclaimed ref nodes.
    pass1_ref_nodes = {m["ref_node"] for m in pass1_matches}

    all_ref_nodes = list(ref_graph.nodes())
    ref_coords = np.array([ref_graph.nodes[rn]["coords"] for rn in all_ref_nodes])
    
    # Vectorized distance computation
    ref_distances = np.linalg.norm(ref_coords - ref_anc_coords, axis=1)
    
    # Select nodes within search radius and not in pass1_matches
    window_ref_nodes = [
        rn for rn, d in zip(all_ref_nodes, ref_distances)
        if rn not in pass1_ref_nodes and d <= search_radius
    ]

    return window_ref_nodes, search_radius, max_query_radius

# =============================================================================
# Pass 2 Cost Matrix and Matching (restricted + relaxed type penalty)
# =============================================================================

def compute_pass2_cost_matrix(
    unmatched_query_nodes,
    window_ref_nodes,
    ref_graph,
    query_graph,
    ref_sigs,
    query_sigs,
    ref_smiles,
    query_smiles
):
    """
    Build a restricted cost matrix for Pass 2.

    Rows    = unmatched query nodes only
    Columns = ref nodes inside the spatial window only
    Type penalty uses the soft 3-tier hierarchy instead of binary TYPE_PENALTY.

    Parameters
    ----------
    unmatched_query_nodes : list of query nodes not matched in Pass 1
    window_ref_nodes      : list of ref nodes inside the spatial window
    ref_graph             : pharmacophore graph of the reference molecule
    query_graph           : pharmacophore graph of the query molecule
    ref_sigs              : precomputed node signatures for ref
    query_sigs            : precomputed node signatures for query

    Returns
    -------
    cost          : (n_unmatched_query x n_window_ref) cost matrix
    pass2_matches : list of match dicts in the same format as Pass 1 matches
    """

    if not unmatched_query_nodes or not window_ref_nodes:
        return np.array([]), []

    ref_mol = Chem.MolFromSmiles(ref_smiles)
    query_mol = Chem.MolFromSmiles(query_smiles)

    # --- Signature cost ---
    ref_mat   = np.array([ref_sigs[rn]   for rn in window_ref_nodes])
    query_mat = np.array([query_sigs[qn] for qn in unmatched_query_nodes])
    sig_cost  = cdist(query_mat, ref_mat, metric="euclidean")
    # shape: (n_unmatched_query, n_window_ref)

    # --- Geometric consistency (VECTORIZED) ---
    # Same formula as Pass 1 but on the restricted node sets
    ref_coords   = np.array([ref_graph.nodes[rn]["coords"]   for rn in window_ref_nodes])
    query_coords = np.array([query_graph.nodes[qn]["coords"] for qn in unmatched_query_nodes])

    D_ref   = cdist(ref_coords,   ref_coords,   metric="euclidean")
    D_query = cdist(query_coords, query_coords, metric="euclidean")

    # Geometric consistency: for each tentative (i,j) pairing, measure how well
    # the pairwise distances from i and j agree with those of the rest of the graph.
    consistency_cost = np.zeros((len(unmatched_query_nodes), len(window_ref_nodes)))
    for i in range(len(unmatched_query_nodes)):
        for j in range(len(window_ref_nodes)):
            best_match = np.abs(D_query[i, :, None] - D_ref[None, j, :]).min(axis=1)
            consistency_cost[i, j] = best_match.mean()

    # --- Soft type penalty (3-tier hierarchy) ---
    # type_cost = np.zeros((len(unmatched_query_nodes), len(window_ref_nodes)))
    # for i, qn in enumerate(unmatched_query_nodes):
    #     for j, rn in enumerate(window_ref_nodes):
    #         type_cost[i, j] = soft_type_penalty(query_graph, qn, ref_graph, rn)
    type_cost = compute_fingerprint_type_cost(window_ref_nodes, unmatched_query_nodes, ref_graph,query_graph, 
                                              ref_mol, query_mol).T
    cost = sig_cost + 0.5 * consistency_cost + type_cost

    # --- Hungarian matching on restricted matrix ---
    row_ind, col_ind = linear_sum_assignment(cost)

    raw_matches = []
    for r, c in zip(row_ind, col_ind):
        # Hard wall: if cost is at or above HARD_TYPE_PENALTY the match is
        # meaningless — different cluster types forced together
        if cost[r, c] >= HARD_TYPE_PENALTY:
            continue

        qn = unmatched_query_nodes[r]
        rn = window_ref_nodes[c]

        raw_matches.append({
            "ref_node":           rn,
            "query_node":         qn,
            "ref_type":           ref_graph.nodes[rn]["ftype"],
            "query_type":         query_graph.nodes[qn]["ftype"],
            "cost":               cost[r, c],
            "ref_atom_indices":   tuple(sorted(ref_graph.nodes[rn].get("atom_indices", []))),
            "query_atom_indices": tuple(sorted(query_graph.nodes[qn].get("atom_indices", []))),
            "pass":               2,   # tag so caller knows where this came from
        })

    raw_matches.sort(key=lambda x: x["cost"])

    # Deduplicate on atom indices — same logic as hungarian_matching
    seen_ref, seen_query = set(), set()
    pass2_matches = []
    for m in raw_matches:
        if (m["ref_atom_indices"] not in seen_ref
                and m["query_atom_indices"] not in seen_query):
            seen_ref.add(m["ref_atom_indices"])
            seen_query.add(m["query_atom_indices"])
            pass2_matches.append(m)

    return cost, pass2_matches


def run_pass2_matching(
    unmatched_query_nodes,
    window_ref_nodes,
    ref_graph,
    query_graph,
    ref_sigs,
    query_sigs,
    ref_smiles,
    query_smiles
):
    """
    Thin wrapper around compute_pass2_cost_matrix.
    Handles the edge case where there is nothing to match.

    Returns pass2_matches (empty list if nothing to do).
    """
    if not unmatched_query_nodes:
        return []   # all query nodes already matched in Pass 1

    if not window_ref_nodes:
        return []   # spatial window is empty — no candidates in ref

    _, pass2_matches = compute_pass2_cost_matrix(
        unmatched_query_nodes,
        window_ref_nodes,
        ref_graph,
        query_graph,
        ref_sigs,
        query_sigs,
        ref_smiles,
        query_smiles
    )
    return pass2_matches

# =============================================================================
# Merge Pass 1 and Pass 2 matches
# =============================================================================

def merge_matches(pass1_matches, pass2_matches):
    """
    Merge Pass 1 and Pass 2 matches into a single deduplicated list.

    Pass 1 matches take priority — if there is any atom index conflict
    between a Pass 2 match and an existing Pass 1 match, the Pass 2
    match is dropped. This preserves the high-confidence spatial anchor
    matches from Pass 1.

    Parameters
    ----------
    pass1_matches : spatially consistent matches from spatial_consistency_filter
    pass2_matches : relaxed-type matches from run_pass2_matching

    Returns
    -------
    merged_matches : combined deduplicated list, Pass 1 first then Pass 2
    """
    seen_ref   = {m["ref_atom_indices"]   for m in pass1_matches}
    seen_query = {m["query_atom_indices"] for m in pass1_matches}

    merged_matches = list(pass1_matches)   # start with all Pass 1

    for m in pass2_matches:
        if (m["ref_atom_indices"]   not in seen_ref
                and m["query_atom_indices"] not in seen_query):
            seen_ref.add(m["ref_atom_indices"])
            seen_query.add(m["query_atom_indices"])
            merged_matches.append(m)

    return merged_matches



# =============================================================================
# Updated full region finding pipeline
# =============================================================================

def find_replaceable_ref_region_v2(
    pipeline_result,
    ref_smiles,
    query_smiles,
    spatial_threshold=2.0,
    spatial_buffer=2.0,
    
):
    """
    Full two-pass pipeline to find the contiguous region of the reference
    molecule that can be replaced by the query scaffold.

    Pass 1 — strict type matching, full ref graph, spatial consistency filter
    Pass 2 — relaxed type hierarchy, restricted spatial window, fills in
              query nodes that had no spatially consistent match in Pass 1

    Bystander ref nodes (inside the window but unmatched) are folded into
    the replaceable region directly without needing gap-fill paths.

    Parameters
    ----------
    pipeline_result    : output of ph_similarity_pipeline (contains raw matches)
    ref_smiles         : SMILES string of the reference molecule
    ref_graph          : pharmacophore graph of reference
    query_graph        : pharmacophore graph of query
    ref_sigs           : precomputed node signatures for ref
    query_sigs         : precomputed node signatures for query
    spatial_threshold  : max |d_ref - d_query| allowed in Pass 1 filter (Angstroms)
    spatial_buffer     : extra radius added on top of max_query_radius (Angstroms)
    max_bond_distance  : passed through to find_replaceable_ref_region stage 2
    outlier_limit      : passed through to find_replaceable_ref_region stage 2

    Returns
    -------
    dict with keys:
        replaceable_atom_indices  — final region including gap-fill + bystanders
        anchor_atom_indices       — clean anchors after outlier pruning
        pass1_matches             — high confidence spatially consistent matches
        pass2_matches             — relaxed type matches within spatial window
        merged_matches            — combined match list fed into region finding
        bystander_atom_indices    — window nodes with no match, included directly
        window_ref_nodes          — all ref nodes considered in Pass 2
        search_radius             — spatial window radius used
        dropped_duplicate_atoms   — from Stage 1 of find_replaceable_ref_region
        dropped_outlier_atoms     — from Stage 2 of find_replaceable_ref_region
        n_anchors_final           — number of anchors after pruning
        coverage_of_ref           — fraction of ref atoms in replaceable region
    """

    # ------------------------------------------------------------------
    # Pass 1: spatial consistency filter on raw Hungarian matches
    # ------------------------------------------------------------------
    raw_matches = pipeline_result.get("matches", [])
    ref_graph   = pipeline_result["ref_graph"]
    ref_sigs    = pipeline_result["ref_sigs"]
    query_graph = pipeline_result["query_graph"]
    query_sigs = pipeline_result["query_sigs"]

    if not raw_matches:
        return {
            "replaceable_atom_indices": [],
            "anchor_atom_indices":      [],
            "pass1_matches":            [],
            "pass2_matches":            [],
            "merged_matches":           [],
            "bystander_atom_indices":   [],
            "window_ref_nodes":         [],
            "search_radius":            0.0,
            "dropped_duplicate_atoms":  [],
            "dropped_outlier_atoms":    [],
            "n_anchors_final":          0,
            "coverage_of_ref":          0.0,
        }

    pass1_matches, unmatched_query_nodes = spatial_consistency_filter(
        raw_matches,
        ref_graph,
        query_graph,
        threshold=spatial_threshold,
    )

    # ------------------------------------------------------------------
    # Pass 2: define spatial window, match unmatched query nodes
    # ------------------------------------------------------------------
    window_ref_nodes, search_radius, _ = define_spatial_window(
        pass1_matches,
        ref_graph,
        query_graph,
        buffer=spatial_buffer,
    )

    pass2_matches = run_pass2_matching(
        unmatched_query_nodes,
        window_ref_nodes,
        ref_graph,
        query_graph,
        ref_sigs,
        query_sigs,
        ref_smiles,
        query_smiles
    )

    # ------------------------------------------------------------------
    # Merge and collect bystanders
    # ------------------------------------------------------------------
    merged_matches = merge_matches(pass1_matches, pass2_matches)

   
    return {        
        "merged_matches":           merged_matches,
        "pass1_matches": pass1_matches,
    }

# =============================================================================
# Aligning the two molecules based on matched nodes
#  > Deduplicating aligned to keep the lowest distance
#  > Filtering single atom index nodes that are having a distance of >2A
#  > Computing if the rmsd improved with the removal of the single atom node
#  > If rmsd improved removing the single atom index match
# =============================================================================



def deduplicated_node_counts(ref_graph, query_graph):
    """
    Count nodes surviving deduplication by atom_indices in both graphs.

    """
    def _dedup(graph):
        seen     = set()
        surviving = []
        for n in graph.nodes():
            fp = frozenset(graph.nodes[n].get("atom_indices", []))
            if fp not in seen:
                seen.add(fp)
                surviving.append(n)
        return surviving

    ref_surviving   = _dedup(ref_graph)
    query_surviving = _dedup(query_graph)

    return {
        "ref_total":      ref_graph.number_of_nodes(),
        "ref_surviving":  len(ref_surviving),
        "ref_nodes":      ref_surviving,
        "query_total":    query_graph.number_of_nodes(),
        "query_surviving":len(query_surviving),
        "query_nodes":    query_surviving,
    }

# def align_and_visualize(
#     pipeline_result,
#     ref_smiles,
#     query_smiles,
#     output_path="alignment.png",
#     show_alignment=True
# ):
#     """
#     Align query onto ref using Kabsch on matched feature centroids.
#     Visualize the overlay and report per-atom mappings.

#     Parameters
#     ----------
#     pipeline_result : result from the pipeline containing ref_graph, query_graph
#     ref_smiles      : SMILES of reference molecule
#     query_smiles    : SMILES of query molecule
#     output_path     : path to save the PNG image
#     show_alignment  : whether to show/save the alignment plot

#     Returns
#     -------
#     R               : 2x2 rotation matrix
#     t               : translation vector
#     atom_mapping    : list of dicts describing per-atom alignment
#     rmsd            : RMSD of centroid alignment
#     query_atom_coords_aligned : aligned query atom coordinates
#     result_dict     : dict with replaceable region info and coverage stats
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     from scipy.optimize import linear_sum_assignment

#     ref_graph  = pipeline_result['ref_graph']
#     query_graph = pipeline_result['query_graph']
#     match_info  = find_replaceable_ref_region_v2(pipeline_result,ref_smiles,query_smiles)
#     merged_matches = match_info['merged_matches']

#     # ------------------------------------------------------------------
#     # Step 1 — Build mol objects and get 2D coords
#     # ------------------------------------------------------------------
#     ref_mol   = Chem.MolFromSmiles(ref_smiles)
#     query_mol = Chem.MolFromSmiles(query_smiles)
#     AllChem.Compute2DCoords(ref_mol)
#     AllChem.Compute2DCoords(query_mol)

#     ref_conf   = ref_mol.GetConformer()
#     query_conf = query_mol.GetConformer()

#     def get_atom_coords(conf, idx):
#         pos = conf.GetAtomPosition(idx)
#         return np.array([pos.x, pos.y])

#     ref_atom_coords   = {i: get_atom_coords(ref_conf, i)
#                          for i in range(ref_mol.GetNumAtoms())}
#     query_atom_coords = {i: get_atom_coords(query_conf, i)
#                          for i in range(query_mol.GetNumAtoms())}

#     # ------------------------------------------------------------------
#     # Step 2 — Kabsch on feature centroids
#     # ------------------------------------------------------------------
#     ref_centroids   = np.array([
#         ref_graph.nodes[m['ref_node']]['coords']
#         for m in merged_matches
#     ])
#     query_centroids = np.array([
#         query_graph.nodes[m['query_node']]['coords']
#         for m in merged_matches
#     ])

#     R, t, query_centroids_aligned = kabsch_2d(query_centroids, ref_centroids)

#     # RMSD on centroids
#     diff = query_centroids_aligned - ref_centroids
#     rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))

#     # ------------------------------------------------------------------
#     # Step 2b — Outlier removal for single-atom matches post-alignment
#     # ------------------------------------------------------------------
#     OUTLIER_DIST_THRESHOLD = 2.0
#     OUTLIER_RMSD_GAIN_MIN  = 0.3

#     def compute_rmsd(q_cents, r_cents):
#         R_, t_, q_aligned_ = kabsch_2d(q_cents, r_cents)
#         diff_ = q_aligned_ - r_cents
#         rmsd_ = float(np.sqrt((diff_ ** 2).sum(axis=1).mean()))
#         return R_, t_, q_aligned_, rmsd_

#     def single_atom_dist_after_align(match, R_, t_, ref_graph_, query_graph_):
#         ref_atoms_   = list(match['ref_atom_indices'])
#         query_atoms_ = list(match['query_atom_indices'])
#         if len(ref_atoms_) != 1 or len(query_atoms_) != 1:
#             return None
#         q_coord = query_graph_.nodes[match['query_node']]['coords']
#         r_coord = ref_graph_.nodes[match['ref_node']]['coords']
#         q_aligned_coord = q_coord @ R_.T + t_
#         return float(np.linalg.norm(q_aligned_coord - r_coord))

#     removed_outliers = []
#     active_matches   = list(merged_matches)

#     improved = True
#     while improved and len(active_matches) > 2:
#         improved = False

#         r_cents = np.array([ref_graph.nodes[m['ref_node']]['coords']
#                              for m in active_matches])
#         q_cents = np.array([query_graph.nodes[m['query_node']]['coords']
#                              for m in active_matches])
#         R, t, query_centroids_aligned, rmsd = compute_rmsd(q_cents, r_cents)

#         candidate_outliers = []
#         for i, m in enumerate(active_matches):
#             d = single_atom_dist_after_align(m, R, t, ref_graph, query_graph)
#             if d is not None and d > OUTLIER_DIST_THRESHOLD:
#                 candidate_outliers.append((i, d))

#         if not candidate_outliers:
#             break

#         best_gain      = 0.0
#         best_rm_idx    = None
#         best_R, best_t = R, t
#         best_q_aligned = query_centroids_aligned
#         best_rmsd_new  = rmsd

#         for rm_i, dist in candidate_outliers:
#             trial_matches = [m for k, m in enumerate(active_matches) if k != rm_i]
#             if len(trial_matches) < 2:
#                 continue
#             r_c = np.array([ref_graph.nodes[m['ref_node']]['coords']
#                              for m in trial_matches])
#             q_c = np.array([query_graph.nodes[m['query_node']]['coords']
#                              for m in trial_matches])
#             R_, t_, q_al_, rmsd_ = compute_rmsd(q_c, r_c)
#             gain = rmsd - rmsd_
#             if gain >= OUTLIER_RMSD_GAIN_MIN and gain > best_gain:
#                 best_gain      = gain
#                 best_rm_idx    = rm_i
#                 best_R, best_t = R_, t_
#                 best_q_aligned = q_al_
#                 best_rmsd_new  = rmsd_

#         if best_rm_idx is not None:
#             outlier = active_matches.pop(best_rm_idx)
#             removed_outliers.append({
#                 'match':              outlier,
#                 'dist_pre_removal':   dict(candidate_outliers)[best_rm_idx],
#                 'rmsd_before':        rmsd,
#                 'rmsd_after':         best_rmsd_new,
#             })
#             R, t                    = best_R, best_t
#             query_centroids_aligned = best_q_aligned
#             rmsd                    = best_rmsd_new
#             improved                = True

#     # Refresh centroids arrays to match the (possibly pruned) active_matches
#     merged_matches = active_matches
#     ref_centroids  = np.array([ref_graph.nodes[m['ref_node']]['coords']
#                                 for m in merged_matches])

#     # ------------------------------------------------------------------
#     # Step 3 — Transform ALL query atom coords into ref-space
#     # ------------------------------------------------------------------
#     query_atom_coords_aligned = {
#         i: coords @ R.T + t
#         for i, coords in query_atom_coords.items()
#     }

#     # ------------------------------------------------------------------
#     # Step 4a — Per-atom mapping within each match (Hungarian)
#     # ------------------------------------------------------------------
#     atom_mapping = []

#     for m in merged_matches:
#         ref_atoms   = list(m['ref_atom_indices'])
#         query_atoms = list(m['query_atom_indices'])

#         if len(ref_atoms) == 1 and len(query_atoms) == 1:
#             r_idx = ref_atoms[0]
#             q_idx = query_atoms[0]
#             dist  = np.linalg.norm(
#                 query_atom_coords_aligned[q_idx] - ref_atom_coords[r_idx]
#             )
#             atom_mapping.append({
#                 'ref_node':   m['ref_node'],
#                 'query_node': m['query_node'],
#                 'ref_type':   m['ref_type'],
#                 'query_type': m['query_type'],
#                 'match_pass': m.get('pass', 1),
#                 'atom_pairs': [(r_idx, q_idx, round(dist, 3))],
#             })

#         else:
#             cost_mat = np.array([
#                 [
#                     np.linalg.norm(
#                         query_atom_coords_aligned[q] - ref_atom_coords[r]
#                     )
#                     for q in query_atoms
#                 ]
#                 for r in ref_atoms
#             ])
#             n      = max(len(ref_atoms), len(query_atoms))
#             padded = np.full((n, n), 999.0)
#             padded[:cost_mat.shape[0], :cost_mat.shape[1]] = cost_mat
#             row_ind, col_ind = linear_sum_assignment(padded)

#             atom_pairs = []
#             for r_i, q_i in zip(row_ind, col_ind):
#                 if r_i < len(ref_atoms) and q_i < len(query_atoms):
#                     r_idx = ref_atoms[r_i]
#                     q_idx = query_atoms[q_i]
#                     dist  = cost_mat[r_i, q_i]
#                     atom_pairs.append((r_idx, q_idx, round(dist, 3)))

#             atom_mapping.append({
#                 'ref_node':   m['ref_node'],
#                 'query_node': m['query_node'],
#                 'ref_type':   m['ref_type'],
#                 'query_type': m['query_type'],
#                 'match_pass': m.get('pass', 1),
#                 'atom_pairs': atom_pairs,
#             })

#     # ------------------------------------------------------------------
#     # Step 4b — Deduplicate atom pairs across all matches
#     # ------------------------------------------------------------------
#     all_pairs = []
#     for match_idx, entry in enumerate(atom_mapping):
#         for r_idx, q_idx, dist in entry["atom_pairs"]:
#             all_pairs.append((match_idx, r_idx, q_idx, dist))

#     all_pairs.sort(key=lambda x: x[3])

#     seen_ref   = set()
#     seen_query = set()
#     kept_pairs = []

#     for match_idx, r_idx, q_idx, dist in all_pairs:
#         if r_idx in seen_ref or q_idx in seen_query:
#             continue
#         seen_ref.add(r_idx)
#         seen_query.add(q_idx)
#         kept_pairs.append((match_idx, r_idx, q_idx, dist))

#     for entry in atom_mapping:
#         entry["atom_pairs"] = []

#     for match_idx, r_idx, q_idx, dist in kept_pairs:
#         atom_mapping[match_idx]["atom_pairs"].append((r_idx, q_idx, dist))

#     atom_mapping = [e for e in atom_mapping if e["atom_pairs"]]

#     # ------------------------------------------------------------------
#     # Step 4c — Define replaceable region from deduplicated atom pairs
#     # ------------------------------------------------------------------
#     matched_ref_atoms = set()
#     for entry in atom_mapping:
#         for r_idx, q_idx, dist in entry["atom_pairs"]:
#             matched_ref_atoms.add(r_idx)

#     ref_mol_tmp = Chem.MolFromSmiles(ref_smiles)
#     AllChem.Compute2DCoords(ref_mol_tmp)
#     n_ref = ref_mol_tmp.GetNumAtoms()

#     bond_graph = nx.Graph()
#     bond_graph.add_nodes_from(range(n_ref))
#     for bond in ref_mol_tmp.GetBonds():
#         bond_graph.add_edge(
#             bond.GetBeginAtomIdx(),
#             bond.GetEndAtomIdx()
#         )

#     # ── FIX 1: Connectivity filter ────────────────────────────────────────
#     if len(matched_ref_atoms) > 1:
#         matched_subgraph = bond_graph.subgraph(matched_ref_atoms)
#         if not nx.is_connected(matched_subgraph):
#             components = list(nx.connected_components(matched_subgraph))
#             largest    = max(components, key=len)
#             removed    = matched_ref_atoms - largest
#             print(f"  [Connectivity filter] Matched ref atoms form "
#                   f"{len(components)} disconnected components.")
#             print(f"  Removed atoms: {sorted(removed)} → kept: {sorted(largest)}")

#             for entry in atom_mapping:
#                 entry["atom_pairs"] = [
#                     (r, q, d) for r, q, d in entry["atom_pairs"]
#                     if r in largest
#                 ]
#             atom_mapping      = [e for e in atom_mapping if e["atom_pairs"]]
#             matched_ref_atoms = largest

#     # ── Sync merged_matches to atom_mapping after connectivity filter ─────
#     surviving_ref_nodes     = {entry['ref_node'] for entry in atom_mapping}
#     merged_matches          = [
#         m for m in merged_matches
#         if m['ref_node'] in surviving_ref_nodes
#     ]
#     ref_centroids           = np.array([
#         ref_graph.nodes[m['ref_node']]['coords']
#         for m in merged_matches
#     ])
#     query_centroids_aligned = np.array([
#         query_graph.nodes[m['query_node']]['coords'] @ R.T + t
#         for m in merged_matches
#     ])
#     print(f"  Merged matches after sync : {len(merged_matches)} "
#           f"(ref_nodes: {[m['ref_node'] for m in merged_matches]})")

#     # Gap fill — walk shortest paths between all matched ref atom pairs
#     replaceable_set  = set(matched_ref_atoms)
#     matched_ref_list = sorted(matched_ref_atoms)

#     for i in range(len(matched_ref_list)):
#         for j in range(i + 1, len(matched_ref_list)):
#             try:
#                 path = nx.shortest_path(
#                     bond_graph,
#                     matched_ref_list[i],
#                     matched_ref_list[j]
#                 )
#                 replaceable_set.update(path)
#             except nx.NetworkXNoPath:
#                 pass

#     # ── FIX 2: Size-ratio cap ─────────────────────────────────────────────
#     n_query        = query_mol.GetNumAtoms()
#     MAX_EXCESS     = n_query + 3
#     gap_fill_count = len(replaceable_set) - len(matched_ref_atoms)

#     if gap_fill_count > MAX_EXCESS:
#         print(f"  [Size-ratio cap] Gap-fill added {gap_fill_count} atoms "
#               f"(query has {n_query}, threshold={MAX_EXCESS}) — "
#               f"reverting to direct matches only.")
#         replaceable_set = set(matched_ref_atoms)

#     # ── FIX 3: Enclosure fill + Ring completion ───────────────────────
#     ring_info_ref = ref_mol_tmp.GetRingInfo()

#     changed = True
#     while changed:
#         changed = False

#         # 3a: Enclosure fill — atom fully surrounded by replaceable atoms
#         non_replaceable = set(range(n_ref)) - replaceable_set
#         for atom_idx in list(non_replaceable):
#             atom_neighbors = set(bond_graph.neighbors(atom_idx))
#             if atom_neighbors and atom_neighbors.issubset(replaceable_set):
#                 replaceable_set.add(atom_idx)
#                 changed = True
#                 print(f"  [Enclosure fill] Pulled enclosed ref atom {atom_idx}")

#         # 3b: Ring completion — if a ring is majority replaceable, 
#         # pull in the whole ring
#         for ring in ring_info_ref.AtomRings():
#             ring_set = set(ring)
#             overlap  = ring_set & replaceable_set
#             # If more than half the ring is already replaceable, 
#             # complete the whole ring
#             if 0 < len(overlap) < len(ring_set):
#                 if len(overlap) > len(ring_set) // 2:
#                     before = len(replaceable_set)
#                     replaceable_set.update(ring_set)
#                     if len(replaceable_set) > before:
#                         changed = True
#                         added = ring_set - overlap
#                         print(f"  [Ring completion] Completed ring {sorted(ring_set)}, "
#                             f"added atoms {sorted(added)}")
#     # ─────────────────────────────────────────────────────────────────

#     # ── FIX 4: Fusion-neighbor ring completion ───────────────────────────
#     # If a ref ring's matched atoms are ONLY the fusion-bond atoms
#     # (i.e. every matched atom in this ring also belongs to another ring
#     # that is already fully inside replaceable_set), then the ring has
#     # no independently matched atoms of its own — it is purely a fused
#     # neighbor. Pull the entire ring into the replaceable set so its
#     # atoms don't dangle as orphan fragments.
#     #
#     # Example:
#     #   ring [6,7,8,9,24,25] fully matched → already replaceable
#     #   ring [3,4,6,25,26]   overlap={6,25} — both 6 and 25 are in the
#     #                         fully-matched ring above → fusion neighbor
#     #                         → pull [3,4,26] in as well
#     # ─────────────────────────────────────────────────────────────────────

#     fully_replaceable_rings = [
#         set(ring) for ring in ring_info_ref.AtomRings()
#         if set(ring).issubset(replaceable_set)
#     ]

#     for ring in ring_info_ref.AtomRings():
#         ring_set = set(ring)

#         # Skip rings already fully inside replaceable
#         if ring_set.issubset(replaceable_set):
#             continue

#         overlap = ring_set & replaceable_set
#         if not overlap:
#             continue   # no matched atoms at all — unrelated ring, skip

#         # Check if every overlapping atom belongs to at least one
#         # fully-replaceable ring → all overlaps are fusion-bond atoms
#         all_overlap_are_fusion = all(
#             any(atom in fr for fr in fully_replaceable_rings)
#             for atom in overlap
#         )

#         if all_overlap_are_fusion:
#             added = ring_set - replaceable_set
#             replaceable_set.update(ring_set)
#             print(f"  [Fusion neighbor] Ring {sorted(ring_set)} is a fused neighbor "
#                   f"— pulled in atoms {sorted(added)}")

#     # Refresh replaceable list after fusion neighbor completion

#     replaceable = sorted(replaceable_set)

#     # Find boundary atoms — in replaceable but bonded to atoms outside
#     boundary_atoms = []
#     for atom_idx in replaceable:
#         atom = ref_mol_tmp.GetAtomWithIdx(atom_idx)
#         for neighbor in atom.GetNeighbors():
#             if neighbor.GetIdx() not in replaceable_set:
#                 boundary_atoms.append(atom_idx)
#                 break

#     # Build boundary atom → query atom map
#     ref_to_query = {}
#     for entry in atom_mapping:
#         for r_idx, q_idx, dist in entry["atom_pairs"]:
#             ref_to_query[r_idx] = q_idx

#     boundary_to_query_map = {}
#     for r in boundary_atoms:
#         if r in ref_to_query:
#             boundary_to_query_map[r] = ref_to_query[r]
#         else:
#             # Gap-fill boundary atom — proxy through nearest matched ref atom
#             best_r, best_len = None, float("inf")
#             for matched_r in matched_ref_atoms:
#                 try:
#                     d = nx.shortest_path_length(bond_graph, r, matched_r)
#                     if d < best_len:
#                         best_len, best_r = d, matched_r
#                 except nx.NetworkXNoPath:
#                     pass
#             if best_r is not None:
#                 boundary_to_query_map[r] = ref_to_query[best_r]
#                 print(f"  Gap-fill boundary atom {r} proxied → "
#                       f"ref matched atom {best_r} → "
#                       f"query atom {ref_to_query[best_r]}")
#             else:
#                 print(f"  WARNING: gap-fill boundary atom {r} "
#                       f"has no reachable matched atom — skipped")

#     # ------------------------------------------------------------------
#     # Step 5 — Visualization
#     # ------------------------------------------------------------------
#     if show_alignment:
#         fig, ax = plt.subplots(1, 1, figsize=(14, 8))
#         ax.set_aspect('equal')
#         ax.axis('off')
#         ax.set_title(f"Pharmacophore Alignment  |  RMSD = {rmsd:.3f} Å  |  "
#                      f"N matched features = {len(merged_matches)}", fontsize=12)

#         # Draw ref bonds
#         for bond in ref_mol.GetBonds():
#             i, j   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#             ci, cj = ref_atom_coords[i], ref_atom_coords[j]
#             ax.plot([ci[0], cj[0]], [ci[1], cj[1]],
#                     color='#444444', linewidth=1.5, zorder=1)

#         # Draw query bonds (aligned)
#         for bond in query_mol.GetBonds():
#             i, j   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#             ci, cj = (query_atom_coords_aligned[i],
#                       query_atom_coords_aligned[j])
#             ax.plot([ci[0], cj[0]], [ci[1], cj[1]],
#                     color='#1565C0', linewidth=1.5, zorder=1, linestyle='--')

#         # Draw ref atoms
#         for i, coords in ref_atom_coords.items():
#             symbol = ref_mol.GetAtomWithIdx(i).GetSymbol()
#             ax.scatter(*coords, s=120, color='#EEEEEE',
#                        edgecolors='#444444', zorder=3, linewidth=1)
#             ax.text(coords[0], coords[1] + 0.15, f"{symbol}{i}",
#                     ha='center', va='bottom', fontsize=6,
#                     color='#222222', zorder=4)

#         # Draw query atoms (aligned)
#         for i, coords in query_atom_coords_aligned.items():
#             symbol = query_mol.GetAtomWithIdx(i).GetSymbol()
#             ax.scatter(*coords, s=120, color='#BBDEFB',
#                        edgecolors='#1565C0', zorder=3, linewidth=1)
#             ax.text(coords[0], coords[1] + 0.15, f"{symbol}{i}",
#                     ha='center', va='bottom', fontsize=6,
#                     color='#1565C0', zorder=4)

#         # Draw match lines between aligned centroids
#         for idx, m in enumerate(merged_matches):
#             r_centroid = ref_graph.nodes[m['ref_node']]['coords']
#             q_centroid = query_graph.nodes[m['query_node']]['coords'] @ R.T + t
#             match_pass = m.get('pass', 1)
#             line_color = '#2E7D32' if match_pass == 1 else '#F57F17'

#             ax.plot([r_centroid[0], q_centroid[0]],
#                     [r_centroid[1], q_centroid[1]],
#                     color=line_color, linewidth=1.2,
#                     linestyle=':', zorder=2, alpha=0.8)

#             mid = (r_centroid + q_centroid) / 2
#             ax.text(mid[0], mid[1],
#                     f"{m['ref_type']}↔{m['query_type']}\n"
#                     f"cost={m['cost']:.2f}",
#                     ha='center', va='center', fontsize=5,
#                     color=line_color,
#                     bbox=dict(boxstyle='round,pad=0.1',
#                               facecolor='white', alpha=0.6, linewidth=0),
#                     zorder=5)

#         # Draw atom pair lines
#         for entry in atom_mapping:
#             for r_idx, q_idx, dist in entry['atom_pairs']:
#                 rc = ref_atom_coords[r_idx]
#                 qc = query_atom_coords_aligned[q_idx]
#                 ax.annotate('', xy=rc, xytext=qc,
#                             arrowprops=dict(
#                                 arrowstyle='-',
#                                 color='#E53935',
#                                 lw=0.8,
#                                 alpha=0.5,
#                             ), zorder=2)

#         # Legend
#         legend_elements = [
#             mpatches.Patch(facecolor='#EEEEEE', edgecolor='#444444',
#                            label='Ref atoms'),
#             mpatches.Patch(facecolor='#BBDEFB', edgecolor='#1565C0',
#                            label='Query atoms (aligned)'),
#             plt.Line2D([0], [0], color='#2E7D32', linestyle=':',
#                        label='Pass 1 feature match'),
#             plt.Line2D([0], [0], color='#F57F17', linestyle=':',
#                        label='Pass 2 feature match'),
#             plt.Line2D([0], [0], color='#E53935', linestyle='-',
#                        label='Atom-level pair'),
#         ]
#         ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

#         plt.tight_layout()
#         plt.savefig(output_path, dpi=150, bbox_inches='tight')
#         plt.show()
#         print(f"Saved to {output_path}")

#     # ------------------------------------------------------------------
#     # Step 6 — Print atom mapping report
#     # ------------------------------------------------------------------
#     print("\n" + "=" * 60)
#     print("ATOM ALIGNMENT REPORT (deduplicated)")
#     print("=" * 60)
#     for entry in atom_mapping:
#         if not entry["atom_pairs"]:
#             continue
#         print(f"\n  Feature match: ref_node={entry['ref_node']} "
#               f"({entry['ref_type']}) ↔ "
#               f"query_node={entry['query_node']} "
#               f"({entry['query_type']})  "
#               f"[Pass {entry['match_pass']}]")
#         for r_idx, q_idx, dist in entry["atom_pairs"]:
#             r_sym = ref_mol.GetAtomWithIdx(r_idx).GetSymbol()
#             q_sym = query_mol.GetAtomWithIdx(q_idx).GetSymbol()
#             print(f"    ref atom {r_idx:3d} ({r_sym}) "
#                   f"↔ query atom {q_idx:3d} ({q_sym})  "
#                   f"dist = {dist:.3f} Å")

#     print(f"\n  Overall centroid RMSD: {rmsd:.4f} Å")

#     n_nodes        = deduplicated_node_counts(ref_graph, query_graph)
#     ref_coverage   = len(merged_matches) / n_nodes['ref_surviving']
#     query_coverage = len(merged_matches) / n_nodes['query_surviving']

#     print("\n" + "=" * 60)
#     print("REPLACEABLE REGION")
#     print("=" * 60)
#     print(f"  Matched ref atoms  : {sorted(matched_ref_atoms)}")
#     print(f"  Replaceable atoms  : {replaceable}")
#     print(f"  Boundary atoms     : {boundary_atoms}")
#     print(f"  Boundary→Query map : {boundary_to_query_map}")
#     print(f"  Coverage of Query  : {query_coverage:.2%}, "
#           f"Total Nodes: {n_nodes['query_surviving']}, "
#           f"Matched: {len(merged_matches)}")
#     print("=" * 60)

#     ref_coverage_final   = len(merged_matches) / ref_graph.number_of_nodes()
#     query_coverage_final = len(merged_matches) / query_graph.number_of_nodes()

#     return R, t, atom_mapping, rmsd, query_atom_coords_aligned, {
#         "RMSD":                     f"{rmsd:.4f}",
#         "ref_coverage":             f"{ref_coverage_final:.2%}",
#         "query_coverage":           f"{query_coverage_final:.2%}",
#         "replaceable_atom_indices": replaceable,
#         "boundary_atoms":           boundary_atoms,
#         "boundary_to_query_map":    boundary_to_query_map,
#         "matched_ref_atoms":        sorted(matched_ref_atoms),
#         "coverage_of_ref":          len(replaceable) / n_ref,
#         "query_coords_aligned":     query_atom_coords_aligned,
#         "surviving_matches":        merged_matches,
#         "removed_outlier_matches":  [o["match"] for o in removed_outliers],
#     }
# #
def align_and_visualize(
    pipeline_result,
    ref_smiles,
    query_smiles,
    output_path="alignment.png",
    show_alignment=True,
    hop_buffer=1,
    cost_threshold=3.0,
):
    """
    Align query onto ref using Kabsch on matched feature centroids.
    Visualize the overlay and report per-atom mappings.

    Parameters
    ----------
    pipeline_result  : result from the pipeline containing ref_graph, query_graph
    ref_smiles       : SMILES of reference molecule
    query_smiles     : SMILES of query molecule
    output_path      : path to save the PNG image
    show_alignment   : whether to show/save the alignment plot
    hop_buffer       : max allowed |ref_hops - query_hops| from anchor (default 2)
    cost_threshold   : max match cost to survive connectivity filter (default 2.0)

    Returns
    -------
    R               : 2x2 rotation matrix
    t               : translation vector
    atom_mapping    : list of dicts describing per-atom alignment
    rmsd            : RMSD of centroid alignment
    query_atom_coords_aligned : aligned query atom coordinates
    result_dict     : dict with replaceable region info and coverage stats
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.optimize import linear_sum_assignment

    ref_graph   = pipeline_result['ref_graph']
    query_graph = pipeline_result['query_graph']
    match_info  = find_replaceable_ref_region_v2(pipeline_result, ref_smiles, query_smiles)
    merged_matches  = match_info['merged_matches']
    pass1_matches   = match_info['pass1_matches']

    # ------------------------------------------------------------------
    # Step 1 — Build mol objects and get 2D coords
    # ------------------------------------------------------------------
    ref_mol   = Chem.MolFromSmiles(ref_smiles)
    query_mol = Chem.MolFromSmiles(query_smiles)
    AllChem.Compute2DCoords(ref_mol)
    AllChem.Compute2DCoords(query_mol)

    ref_conf   = ref_mol.GetConformer()
    query_conf = query_mol.GetConformer()

    def get_atom_coords(conf, idx):
        pos = conf.GetAtomPosition(idx)
        return np.array([pos.x, pos.y])

    ref_atom_coords   = {i: get_atom_coords(ref_conf, i)
                         for i in range(ref_mol.GetNumAtoms())}
    query_atom_coords = {i: get_atom_coords(query_conf, i)
                         for i in range(query_mol.GetNumAtoms())}

    # ------------------------------------------------------------------
    # Step 2 — Kabsch on feature centroids
    # ------------------------------------------------------------------
    ref_centroids   = np.array([
        ref_graph.nodes[m['ref_node']]['coords']
        for m in merged_matches
    ])
    query_centroids = np.array([
        query_graph.nodes[m['query_node']]['coords']
        for m in merged_matches
    ])

    R, t, query_centroids_aligned = kabsch_2d(query_centroids, ref_centroids)

    # RMSD on centroids
    diff = query_centroids_aligned - ref_centroids
    rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    # ------------------------------------------------------------------
    # Step 2b — Outlier removal for single-atom matches post-alignment
    # ------------------------------------------------------------------
    OUTLIER_DIST_THRESHOLD = 2.0
    OUTLIER_RMSD_GAIN_MIN  = 0.3

    def compute_rmsd(q_cents, r_cents):
        R_, t_, q_aligned_ = kabsch_2d(q_cents, r_cents)
        diff_ = q_aligned_ - r_cents
        rmsd_ = float(np.sqrt((diff_ ** 2).sum(axis=1).mean()))
        return R_, t_, q_aligned_, rmsd_

    def single_atom_dist_after_align(match, R_, t_, ref_graph_, query_graph_):
        ref_atoms_   = list(match['ref_atom_indices'])
        query_atoms_ = list(match['query_atom_indices'])
        if len(ref_atoms_) != 1 or len(query_atoms_) != 1:
            return None
        q_coord = query_graph_.nodes[match['query_node']]['coords']
        r_coord = ref_graph_.nodes[match['ref_node']]['coords']
        q_aligned_coord = q_coord @ R_.T + t_
        return float(np.linalg.norm(q_aligned_coord - r_coord))

    removed_outliers = []
    active_matches   = list(merged_matches)

    improved = True
    while improved and len(active_matches) > 2:
        improved = False

        r_cents = np.array([ref_graph.nodes[m['ref_node']]['coords']
                             for m in active_matches])
        q_cents = np.array([query_graph.nodes[m['query_node']]['coords']
                             for m in active_matches])
        R, t, query_centroids_aligned, rmsd = compute_rmsd(q_cents, r_cents)

        candidate_outliers = []
        for i, m in enumerate(active_matches):
            d = single_atom_dist_after_align(m, R, t, ref_graph, query_graph)
            if d is not None and d > OUTLIER_DIST_THRESHOLD:
                candidate_outliers.append((i, d))

        if not candidate_outliers:
            break

        best_gain      = 0.0
        best_rm_idx    = None
        best_R, best_t = R, t
        best_q_aligned = query_centroids_aligned
        best_rmsd_new  = rmsd

        for rm_i, dist in candidate_outliers:
            trial_matches = [m for k, m in enumerate(active_matches) if k != rm_i]
            if len(trial_matches) < 2:
                continue
            r_c = np.array([ref_graph.nodes[m['ref_node']]['coords']
                             for m in trial_matches])
            q_c = np.array([query_graph.nodes[m['query_node']]['coords']
                             for m in trial_matches])
            R_, t_, q_al_, rmsd_ = compute_rmsd(q_c, r_c)
            gain = rmsd - rmsd_
            if gain >= OUTLIER_RMSD_GAIN_MIN and gain > best_gain:
                best_gain      = gain
                best_rm_idx    = rm_i
                best_R, best_t = R_, t_
                best_q_aligned = q_al_
                best_rmsd_new  = rmsd_

        if best_rm_idx is not None:
            outlier = active_matches.pop(best_rm_idx)
            removed_outliers.append({
                'match':              outlier,
                'dist_pre_removal':   dict(candidate_outliers)[best_rm_idx],
                'rmsd_before':        rmsd,
                'rmsd_after':         best_rmsd_new,
            })
            R, t                    = best_R, best_t
            query_centroids_aligned = best_q_aligned
            rmsd                    = best_rmsd_new
            improved                = True

    # Refresh centroids arrays to match the (possibly pruned) active_matches
    merged_matches = active_matches
    ref_centroids  = np.array([ref_graph.nodes[m['ref_node']]['coords']
                                for m in merged_matches])

    # ------------------------------------------------------------------
    # Step 3 — Transform ALL query atom coords into ref-space
    # ------------------------------------------------------------------
    query_atom_coords_aligned = {
        i: coords @ R.T + t
        for i, coords in query_atom_coords.items()
    }

    # ------------------------------------------------------------------
    # Step 4a — Per-atom mapping within each match (Hungarian)
    # ------------------------------------------------------------------
    atom_mapping = []

    for m in merged_matches:
        ref_atoms   = list(m['ref_atom_indices'])
        query_atoms = list(m['query_atom_indices'])

        if len(ref_atoms) == 1 and len(query_atoms) == 1:
            r_idx = ref_atoms[0]
            q_idx = query_atoms[0]
            dist  = np.linalg.norm(
                query_atom_coords_aligned[q_idx] - ref_atom_coords[r_idx]
            )
            atom_mapping.append({
                'ref_node':   m['ref_node'],
                'query_node': m['query_node'],
                'ref_type':   m['ref_type'],
                'query_type': m['query_type'],
                'match_pass': m.get('pass', 1),
                'cost':       m.get('cost', float('inf')),
                'atom_pairs': [(r_idx, q_idx, round(dist, 3))],
            })

        else:
            cost_mat = np.array([
                [
                    np.linalg.norm(
                        query_atom_coords_aligned[q] - ref_atom_coords[r]
                    )
                    for q in query_atoms
                ]
                for r in ref_atoms
            ])
            n      = max(len(ref_atoms), len(query_atoms))
            padded = np.full((n, n), 999.0)
            padded[:cost_mat.shape[0], :cost_mat.shape[1]] = cost_mat
            row_ind, col_ind = linear_sum_assignment(padded)

            atom_pairs = []
            for r_i, q_i in zip(row_ind, col_ind):
                if r_i < len(ref_atoms) and q_i < len(query_atoms):
                    r_idx = ref_atoms[r_i]
                    q_idx = query_atoms[q_i]
                    dist  = cost_mat[r_i, q_i]
                    atom_pairs.append((r_idx, q_idx, round(dist, 3)))

            atom_mapping.append({
                'ref_node':   m['ref_node'],
                'query_node': m['query_node'],
                'ref_type':   m['ref_type'],
                'query_type': m['query_type'],
                'match_pass': m.get('pass', 1),
                'cost':       m.get('cost', float('inf')),
                'atom_pairs': atom_pairs,
            })

    # ------------------------------------------------------------------
    # Step 4b — Deduplicate atom pairs across all matches
    # ------------------------------------------------------------------
    all_pairs = []
    for match_idx, entry in enumerate(atom_mapping):
        for r_idx, q_idx, dist in entry["atom_pairs"]:
            all_pairs.append((match_idx, r_idx, q_idx, dist))

    all_pairs.sort(key=lambda x: x[3])

    seen_ref   = set()
    seen_query = set()
    kept_pairs = []

    for match_idx, r_idx, q_idx, dist in all_pairs:
        if r_idx in seen_ref or q_idx in seen_query:
            continue
        seen_ref.add(r_idx)
        seen_query.add(q_idx)
        kept_pairs.append((match_idx, r_idx, q_idx, dist))

    for entry in atom_mapping:
        entry["atom_pairs"] = []

    for match_idx, r_idx, q_idx, dist in kept_pairs:
        atom_mapping[match_idx]["atom_pairs"].append((r_idx, q_idx, dist))

    atom_mapping = [e for e in atom_mapping if e["atom_pairs"]]

    # ------------------------------------------------------------------
    # Step 4c — Build bond graphs for hop-buffer connectivity filter
    # ------------------------------------------------------------------
    ref_mol_tmp = Chem.MolFromSmiles(ref_smiles)
    AllChem.Compute2DCoords(ref_mol_tmp)
    n_ref = ref_mol_tmp.GetNumAtoms()

    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(range(n_ref))
    for bond in ref_mol_tmp.GetBonds():
        bond_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    query_mol_tmp = Chem.MolFromSmiles(query_smiles)
    query_bond_graph = nx.Graph()
    query_bond_graph.add_nodes_from(range(query_mol_tmp.GetNumAtoms()))
    for bond in query_mol_tmp.GetBonds():
        query_bond_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # ------------------------------------------------------------------
    # Step 4d — Hop-buffer connectivity filter
    # Replace the strict largest-component filter with a more inclusive
    # check: keep a match if its atom pair is within hop_buffer bonds of
    # the anchor in both ref and query spaces, AND cost < cost_threshold.
    # The anchor match is always kept regardless.
    # ------------------------------------------------------------------
    matched_ref_atoms = set()
    for entry in atom_mapping:
        for r_idx, q_idx, dist in entry["atom_pairs"]:
            matched_ref_atoms.add(r_idx)

    if len(matched_ref_atoms) > 1:
        matched_subgraph = bond_graph.subgraph(matched_ref_atoms)
        if not nx.is_connected(matched_subgraph):

            # Anchor: lowest-cost pass1 match — always survives
            anchor_ref_node   = pass1_matches[0]['ref_node']
            anchor_query_node = pass1_matches[0]['query_node']

            anchor_ref_atoms   = ref_graph.nodes[anchor_ref_node].get('atom_indices', [])
            anchor_query_atoms = query_graph.nodes[anchor_query_node].get('atom_indices', [])

            # Use the first atom of the anchor feature as the hop origin
            anchor_ref_atom   = anchor_ref_atoms[0]   if anchor_ref_atoms   else None
            anchor_query_atom = anchor_query_atoms[0] if anchor_query_atoms else None

            surviving_entries = []
            for entry in atom_mapping:

                # Anchor entry always survives
                if entry['ref_node'] == anchor_ref_node:
                    surviving_entries.append(entry)
                    continue

                keep = False
                entry_cost = entry.get('cost', float('inf'))

                # Cost gate — if cost is already too high, skip hop check
                if entry_cost >= cost_threshold:
                    continue

                for r_idx, q_idx, dist in entry["atom_pairs"]:
                    if anchor_ref_atom is None or anchor_query_atom is None:
                        keep = True
                        break

                    try:
                        ref_hops   = nx.shortest_path_length(bond_graph,       anchor_ref_atom,   r_idx)
                        query_hops = nx.shortest_path_length(query_bond_graph, anchor_query_atom, q_idx)
                    except nx.NetworkXNoPath:
                        continue

                    if abs(ref_hops - query_hops) <= hop_buffer:
                        keep = True
                        break

                if keep:
                    surviving_entries.append(entry)

            print(f"  [Hop-buffer filter] {len(atom_mapping)} → "
                  f"{len(surviving_entries)} entries kept "
                  f"(hop_buffer={hop_buffer}, cost_threshold={cost_threshold})")

            atom_mapping      = surviving_entries
            matched_ref_atoms = set()
            for entry in atom_mapping:
                for r_idx, q_idx, dist in entry["atom_pairs"]:
                    matched_ref_atoms.add(r_idx)

    # ------------------------------------------------------------------
    # Sync merged_matches to surviving atom_mapping entries
    # ------------------------------------------------------------------
    surviving_ref_nodes = {entry['ref_node'] for entry in atom_mapping}
    merged_matches = [
        m for m in merged_matches
        if m['ref_node'] in surviving_ref_nodes
    ]
    ref_centroids = np.array([
        ref_graph.nodes[m['ref_node']]['coords']
        for m in merged_matches
    ])
    query_centroids_aligned = np.array([
        query_graph.nodes[m['query_node']]['coords'] @ R.T + t
        for m in merged_matches
    ])
    print(f"  Merged matches after sync : {len(merged_matches)} "
          f"(ref_nodes: {[m['ref_node'] for m in merged_matches]})")

    # ------------------------------------------------------------------
    # Gap fill — walk shortest paths between all matched ref atom pairs
    # ------------------------------------------------------------------
    replaceable_set  = set(matched_ref_atoms)
    matched_ref_list = sorted(matched_ref_atoms)

    for i in range(len(matched_ref_list)):
        for j in range(i + 1, len(matched_ref_list)):
            try:
                path = nx.shortest_path(
                    bond_graph,
                    matched_ref_list[i],
                    matched_ref_list[j]
                )
                replaceable_set.update(path)
            except nx.NetworkXNoPath:
                pass

    # ── Size-ratio cap ────────────────────────────────────────────────
    n_query        = query_mol.GetNumAtoms()
    MAX_EXCESS     = n_query + 3
    gap_fill_count = len(replaceable_set) - len(matched_ref_atoms)

    if gap_fill_count > MAX_EXCESS:
        print(f"  [Size-ratio cap] Gap-fill added {gap_fill_count} atoms "
              f"(query has {n_query}, threshold={MAX_EXCESS}) — "
              f"reverting to direct matches only.")
        replaceable_set = set(matched_ref_atoms)

    # ── Enclosure fill + Ring completion ─────────────────────────────
    ring_info_ref = ref_mol_tmp.GetRingInfo()

    changed = True
    while changed:
        changed = False

        non_replaceable = set(range(n_ref)) - replaceable_set
        for atom_idx in list(non_replaceable):
            atom_neighbors = set(bond_graph.neighbors(atom_idx))
            if atom_neighbors and atom_neighbors.issubset(replaceable_set):
                replaceable_set.add(atom_idx)
                changed = True
                print(f"  [Enclosure fill] Pulled enclosed ref atom {atom_idx}")

        for ring in ring_info_ref.AtomRings():
            ring_set = set(ring)
            overlap  = ring_set & replaceable_set
            if 0 < len(overlap) < len(ring_set):
                if len(overlap) > len(ring_set) // 2:
                    before = len(replaceable_set)
                    replaceable_set.update(ring_set)
                    if len(replaceable_set) > before:
                        changed = True
                        added = ring_set - overlap
                        print(f"  [Ring completion] Completed ring {sorted(ring_set)}, "
                              f"added atoms {sorted(added)}")

    # ── Fusion-neighbor ring completion ──────────────────────────────
    fully_replaceable_rings = [
        set(ring) for ring in ring_info_ref.AtomRings()
        if set(ring).issubset(replaceable_set)
    ]

    for ring in ring_info_ref.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(replaceable_set):
            continue
        overlap = ring_set & replaceable_set
        if not overlap:
            continue
        all_overlap_are_fusion = all(
            any(atom in fr for fr in fully_replaceable_rings)
            for atom in overlap
        )
        if all_overlap_are_fusion:
            added = ring_set - replaceable_set
            replaceable_set.update(ring_set)
            print(f"  [Fusion neighbor] Ring {sorted(ring_set)} is a fused neighbor "
                  f"— pulled in atoms {sorted(added)}")

    replaceable = sorted(replaceable_set)

    

    # Find boundary atoms — in replaceable but bonded to atoms outside
    boundary_atoms = []
    for atom_idx in replaceable:
        atom = ref_mol_tmp.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in replaceable_set:
                boundary_atoms.append(atom_idx)
                break

    # Build boundary atom → query atom map
    ref_to_query = {}
    for entry in atom_mapping:
        for r_idx, q_idx, dist in entry["atom_pairs"]:
            ref_to_query[r_idx] = q_idx

    boundary_to_query_map = {}
    for r in boundary_atoms:
        if r in ref_to_query:
            boundary_to_query_map[r] = ref_to_query[r]
        else:
            best_r, best_len = None, float("inf")
            for matched_r in matched_ref_atoms:
                try:
                    d = nx.shortest_path_length(bond_graph, r, matched_r)
                    if d < best_len:
                        best_len, best_r = d, matched_r
                except nx.NetworkXNoPath:
                    pass
            if best_r is not None:
                boundary_to_query_map[r] = ref_to_query[best_r]
                print(f"  Gap-fill boundary atom {r} proxied → "
                      f"ref matched atom {best_r} → "
                      f"query atom {ref_to_query[best_r]}")
            else:
                print(f"  WARNING: gap-fill boundary atom {r} "
                      f"has no reachable matched atom — skipped")

    # ------------------------------------------------------------------
    # Step 5 — Visualization
    # ------------------------------------------------------------------
    if show_alignment:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Pharmacophore Alignment  |  RMSD = {rmsd:.3f} Å  |  "
                     f"N matched features = {len(merged_matches)}", fontsize=12)

        for bond in ref_mol.GetBonds():
            i, j   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ci, cj = ref_atom_coords[i], ref_atom_coords[j]
            ax.plot([ci[0], cj[0]], [ci[1], cj[1]],
                    color='#444444', linewidth=1.5, zorder=1)

        for bond in query_mol.GetBonds():
            i, j   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ci, cj = (query_atom_coords_aligned[i],
                      query_atom_coords_aligned[j])
            ax.plot([ci[0], cj[0]], [ci[1], cj[1]],
                    color='#1565C0', linewidth=1.5, zorder=1, linestyle='--')

        for i, coords in ref_atom_coords.items():
            symbol = ref_mol.GetAtomWithIdx(i).GetSymbol()
            ax.scatter(*coords, s=120, color='#EEEEEE',
                       edgecolors='#444444', zorder=3, linewidth=1)
            ax.text(coords[0], coords[1] + 0.15, f"{symbol}{i}",
                    ha='center', va='bottom', fontsize=6,
                    color='#222222', zorder=4)

        for i, coords in query_atom_coords_aligned.items():
            symbol = query_mol.GetAtomWithIdx(i).GetSymbol()
            ax.scatter(*coords, s=120, color='#BBDEFB',
                       edgecolors='#1565C0', zorder=3, linewidth=1)
            ax.text(coords[0], coords[1] + 0.15, f"{symbol}{i}",
                    ha='center', va='bottom', fontsize=6,
                    color='#1565C0', zorder=4)

        for idx, m in enumerate(merged_matches):
            r_centroid = ref_graph.nodes[m['ref_node']]['coords']
            q_centroid = query_graph.nodes[m['query_node']]['coords'] @ R.T + t
            match_pass = m.get('pass', 1)
            line_color = '#2E7D32' if match_pass == 1 else '#F57F17'

            ax.plot([r_centroid[0], q_centroid[0]],
                    [r_centroid[1], q_centroid[1]],
                    color=line_color, linewidth=1.2,
                    linestyle=':', zorder=2, alpha=0.8)

            mid = (r_centroid + q_centroid) / 2
            ax.text(mid[0], mid[1],
                    f"{m['ref_type']}↔{m['query_type']}\n"
                    f"cost={m['cost']:.2f}",
                    ha='center', va='center', fontsize=5,
                    color=line_color,
                    bbox=dict(boxstyle='round,pad=0.1',
                              facecolor='white', alpha=0.6, linewidth=0),
                    zorder=5)

        for entry in atom_mapping:
            for r_idx, q_idx, dist in entry['atom_pairs']:
                rc = ref_atom_coords[r_idx]
                qc = query_atom_coords_aligned[q_idx]
                ax.annotate('', xy=rc, xytext=qc,
                            arrowprops=dict(
                                arrowstyle='-',
                                color='#E53935',
                                lw=0.8,
                                alpha=0.5,
                            ), zorder=2)

        legend_elements = [
            mpatches.Patch(facecolor='#EEEEEE', edgecolor='#444444',
                           label='Ref atoms'),
            mpatches.Patch(facecolor='#BBDEFB', edgecolor='#1565C0',
                           label='Query atoms (aligned)'),
            plt.Line2D([0], [0], color='#2E7D32', linestyle=':',
                       label='Pass 1 feature match'),
            plt.Line2D([0], [0], color='#F57F17', linestyle=':',
                       label='Pass 2 feature match'),
            plt.Line2D([0], [0], color='#E53935', linestyle='-',
                       label='Atom-level pair'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {output_path}")

    # ------------------------------------------------------------------
    # Step 6 — Print atom mapping report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ATOM ALIGNMENT REPORT (deduplicated)")
    print("=" * 60)
    for entry in atom_mapping:
        if not entry["atom_pairs"]:
            continue
        print(f"\n  Feature match: ref_node={entry['ref_node']} "
              f"({entry['ref_type']}) ↔ "
              f"query_node={entry['query_node']} "
              f"({entry['query_type']})  "
              f"[Pass {entry['match_pass']}]")
        for r_idx, q_idx, dist in entry["atom_pairs"]:
            r_sym = ref_mol.GetAtomWithIdx(r_idx).GetSymbol()
            q_sym = query_mol.GetAtomWithIdx(q_idx).GetSymbol()
            print(f"    ref atom {r_idx:3d} ({r_sym}) "
                  f"↔ query atom {q_idx:3d} ({q_sym})  "
                  f"dist = {dist:.3f} Å")

    print(f"\n  Overall centroid RMSD: {rmsd:.4f} Å")

    n_nodes        = deduplicated_node_counts(ref_graph, query_graph)
    ref_coverage   = len(merged_matches) / n_nodes['ref_surviving']
    query_coverage = len(merged_matches) / n_nodes['query_surviving']

    print("\n" + "=" * 60)
    print("REPLACEABLE REGION")
    print("=" * 60)
    print(f"  Matched ref atoms  : {sorted(matched_ref_atoms)}")
    print(f"  Replaceable atoms  : {replaceable}")
    print(f"  Boundary atoms     : {boundary_atoms}")
    print(f"  Boundary→Query map : {boundary_to_query_map}")
    print(f"  Coverage of Query  : {query_coverage:.2%}, "
          f"Total Nodes: {n_nodes['query_surviving']}, "
          f"Matched: {len(merged_matches)}")
    print("=" * 60)

    ref_coverage_final   = len(merged_matches) / ref_graph.number_of_nodes()
    query_coverage_final = len(merged_matches) / query_graph.number_of_nodes()

    return R, t, atom_mapping, rmsd, query_atom_coords_aligned, {
        "RMSD":                     f"{rmsd:.4f}",
        "ref_coverage":             f"{ref_coverage_final:.2%}",
        "query_coverage":           f"{query_coverage_final:.2%}",
        "replaceable_atom_indices": replaceable,
        "boundary_atoms":           boundary_atoms,
        "boundary_to_query_map":    boundary_to_query_map,
        "matched_ref_atoms":        sorted(matched_ref_atoms),
        "coverage_of_ref":          len(replaceable) / n_ref,
        "query_coords_aligned":     query_atom_coords_aligned,
        "surviving_matches":        merged_matches,
        "removed_outlier_matches":  [o["match"] for o in removed_outliers],
    }

 # =============================================================================
#


 # =============================================================================
# Extracts molecular fragments outside the replacable region
# =============================================================================

def extract_fragment(
    ref_mol,
    boundary_atom_idx,
    replaceable_set,
):
    """
    Extract fragment subtrees attached to boundary_atom_idx going
    away from the replaceable region.

    Now handles ring fragments — rings that partially overlap with
    the replaceable region. The outside portion of the ring is
    extracted as a chain (ring opened at the boundary) with a flag
    indicating it was a ring so downstream can handle accordingly.
    """
    boundary_atom     = ref_mol.GetAtomWithIdx(boundary_atom_idx)
    outside_neighbors = [
        nb.GetIdx() for nb in boundary_atom.GetNeighbors()
        if nb.GetIdx() not in replaceable_set
    ]

    if not outside_neighbors:
        return [], [], [], False

    ring_info  = ref_mol.GetRingInfo()
    all_rings  = [set(r) for r in ring_info.AtomRings()]

    all_fragment_atoms = []
    root_neighbors     = []
    bond_types         = []
    is_ring_fragment   = False

    for root_neighbor in outside_neighbors:
        # BFS outside replaceable region
        visited = set()
        queue   = [root_neighbor]
        while queue:
            current = queue.pop(0)
            if current in visited or current in replaceable_set:
                continue
            visited.add(current)
            atom = ref_mol.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                if nb.GetIdx() not in visited and nb.GetIdx() not in replaceable_set:
                    queue.append(nb.GetIdx())

        fragment_atoms = sorted(visited)
        frag_set       = set(fragment_atoms)

        # Check for partial ring membership
        partial_ring = False
        for ring in all_rings:
            if ring & frag_set and not ring <= frag_set:
                # Ring partially in fragment, partially in replaceable
                partial_ring      = True
                is_ring_fragment  = True
                break

        bond = ref_mol.GetBondBetweenAtoms(boundary_atom_idx, root_neighbor)

        all_fragment_atoms.append({
            "atoms":        fragment_atoms,
            "is_partial_ring": partial_ring,
        })

        root_neighbors.append(root_neighbor)
        bond_types.append(bond.GetBondType())

    
    return all_fragment_atoms, root_neighbors, bond_types, is_ring_fragment

# =============================================================================
# Find the best atom in a query molecule to attach a fragment to
# =============================================================================

def find_attachment_point(
    new_mol,
    query_attach_atom,
    query_mol_original,
    query_coords_aligned,
    ref_boundary_atom_coord,
    max_neighbor_hops=2,
):
    """
    Find the best available attachment point in query for a fragment.

    Priority:
    1. query_attach_atom itself if it has free valence or implicit Hs
    2. Neighbors of query_attach_atom within max_neighbor_hops
       sorted by distance to ref_boundary_atom in aligned coords

    Parameters
    ----------
    new_mol                  : current RWMol being built
    query_attach_atom        : preferred attachment atom (from boundary map)
    query_mol_original       : original query mol (for topology)
    query_coords_aligned     : {atom_idx: aligned_2d_coords}
    ref_boundary_atom_coord  : 2D coord of the ref boundary atom
    max_neighbor_hops        : how far to search in query bond graph

    Returns
    -------
    attach_idx  : atom index to use for attachment (in new_mol)
    needs_h_removal : bool — True if an H must be removed first
    """
    def has_free_valence(mol, idx):
        atom = mol.GetAtomWithIdx(idx)
        
        # Ensure valence  is up to date before querying implicit Hs
        try:
            atom.UpdatePropertyCache(strict=False)
        except Exception:
            pass  # best-effort; we'll fall through to explicit H check

        # Check implicit Hs — these can be replaced by bonds
        try:
            if atom.GetNumImplicitHs() > 0:
                return True, False
        except RuntimeError:
            pass  # valence still not resolved; fall through

        # Check explicit Hs
        if atom.GetNumExplicitHs() > 0:
            return True, True   # free valence, H removal needed

        return False, False

    # --- Priority 1: preferred atom itself ---
    ok, needs_h = has_free_valence(new_mol, query_attach_atom)
    if ok:
        return query_attach_atom, needs_h

    # --- Priority 2: neighbors within max_neighbor_hops ---
    # BFS over original query topology
    visited  = {query_attach_atom}
    frontier = [query_attach_atom]
    hop      = 0
    candidates = []   # (distance_to_ref_boundary, atom_idx, needs_h_removal)

    while hop < max_neighbor_hops and frontier:
        next_frontier = []
        for current in frontier:
            atom = query_mol_original.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in visited:
                    continue
                visited.add(nb_idx)
                next_frontier.append(nb_idx)

                # Check valence on this neighbor
                ok, needs_h = has_free_valence(new_mol, nb_idx)
                if ok:
                    # Distance from this neighbor to ref boundary atom
                    nb_coord = query_coords_aligned.get(nb_idx)
                    if nb_coord is not None:
                        d = float(np.linalg.norm(nb_coord - ref_boundary_atom_coord))
                    else:
                        d = 999.0
                    candidates.append((d, nb_idx, needs_h))

        frontier = next_frontier
        hop += 1

    if candidates:
        # Pick closest to where the ref boundary atom sits
        candidates.sort(key=lambda x: x[0])
        _, best_idx, needs_h = candidates[0]
        print(f"    Redirected attachment: query atom {query_attach_atom} "
              f"→ neighbor atom {best_idx} "
              f"(closer to ref boundary, dist={candidates[0][0]:.2f}Å)")
        return best_idx, needs_h

    return None, False   # no attachment point found

# =============================================================================
# Builds new molecule by attaching the reference fragments onto the query 
# =============================================================================

# def attach_fragments_to_query(
#     query_mol,
#     ref_mol,
#     boundary_to_query_map,
#     replaceable_set,
#     ref_to_query_atom_map,
#     query_coords_aligned,
# ):
#     from rdkit.Chem import RWMol
#     from collections import defaultdict

#     boundary_to_query_map = {int(k): int(v) for k, v in boundary_to_query_map.items()}
#     replaceable_set       = set(int(a) for a in replaceable_set)
#     ref_to_query_atom_map = {int(k): int(v) for k, v in ref_to_query_atom_map.items()}
#     query_coords_aligned  = {int(k): v for k, v in query_coords_aligned.items()}

#     attachment_log = []
#     failed_log     = []

#     new_mol  = RWMol(Chem.RWMol(query_mol))
#     ref_conf = ref_mol.GetConformer()

#     # Track which atom indices in new_mol were freshly added (not from query_mol)
#     original_atom_count = new_mol.GetNumAtoms()

#     def get_ref_atom_coord(atom_idx):
#         pos = ref_conf.GetAtomPosition(int(atom_idx))
#         return np.array([pos.x, pos.y])

#      # ── NEW helper ──────────────────────────────────────────────────────────
#     def _frag_smiles(frag_atoms_list):
#         """SMILES of the ref fragment atoms (no attachment marker)."""
#         try:
#             smi = Chem.MolFragmentToSmiles(ref_mol, frag_atoms_list)
#             return smi if smi else None
#         except Exception:
#             return None
#     # ────────────────────────────────────────────────────────────────────────
#     # ------------------------------------------------------------------
#     # Step 1: collect all tasks before doing anything
#     # ------------------------------------------------------------------
#     all_tasks = []

#     for ref_boundary_atom, query_attach_atom in boundary_to_query_map.items():
#         ref_boundary_atom = int(ref_boundary_atom)
#         query_attach_atom = int(query_attach_atom)

#         all_fragment_atoms, root_neighbors, bond_types, is_ring_fragment = \
#             extract_fragment(ref_mol, ref_boundary_atom, replaceable_set)

#         if not all_fragment_atoms:
#             failed_log.append({
#                 "ref_boundary_atom": ref_boundary_atom,
#                 "query_attach_atom": query_attach_atom,
#                 "reason": "no fragment outside replaceable region",
#             })
#             continue

#         for frag_info, root_nb, bond_type in zip(
#                 all_fragment_atoms, root_neighbors, bond_types):
#             if not frag_info["atoms"]:
#                 continue
#             all_tasks.append({
#                 "ref_boundary_atom": ref_boundary_atom,
#                 "query_attach_atom": query_attach_atom,
#                 "frag_key":          tuple(sorted(frag_info["atoms"])),
#                 "frag_atoms_list":   frag_info["atoms"],
#                 "is_partial_ring":   frag_info["is_partial_ring"],
#                 "root_nb":           root_nb,
#                 "bond_type":         bond_type,
#             })


#     # ------------------------------------------------------------------
#     # Step 2: group by fragment atom set → detect fused rings
#     # ------------------------------------------------------------------
#     frag_key_to_tasks = defaultdict(list)
#     for task in all_tasks:
#         frag_key_to_tasks[task["frag_key"]].append(task)

#     simple_tasks     = []
#     fused_ring_tasks = []

#     for frag_key, tasks in frag_key_to_tasks.items():
#         partial = [t for t in tasks if t["is_partial_ring"]]
#         normal  = [t for t in tasks if not t["is_partial_ring"]]

#         for t in normal:
#             simple_tasks.append(t)

#         if len(partial) == 2:
#             fused_ring_tasks.append((partial[0], partial[1]))
#             print(f"  FUSED RING DETECTED: fragment {frag_key} shared by "
#                   f"ref boundary atoms {partial[0]['ref_boundary_atom']} and "
#                   f"{partial[1]['ref_boundary_atom']} → "
#                   f"query atoms {partial[0]['query_attach_atom']} and "
#                   f"{partial[1]['query_attach_atom']}")
#         elif len(partial) == 1:
#             simple_tasks.append(partial[0])
#         elif len(partial) > 2:
#             fused_ring_tasks.append((partial[0], partial[1]))
#             for t in partial[2:]:
#                 simple_tasks.append(t)

#         # ------------------------------------------------------------------
#     # Step 2c: Filter out fragments already present in query mol
#     # ------------------------------------------------------------------
#     def _fragment_already_present(frag_atoms_list, query_attach_atom):
#         frag_smi = _frag_smiles(frag_atoms_list)
#         if not frag_smi:
#             return False
#         frag_mol = Chem.MolFromSmiles(frag_smi)
#         if frag_mol is None:
#             return False
#         if not query_mol.HasSubstructMatch(frag_mol):
#             return False
#         matches = query_mol.GetSubstructMatches(frag_mol)
#         attach_neighbors = {
#             n.GetIdx()
#             for n in query_mol.GetAtomWithIdx(query_attach_atom).GetNeighbors()
#         }
#         attach_neighbors.add(query_attach_atom)
#         for match in matches:
#             if any(idx in attach_neighbors for idx in match):
#                 print(f"  SKIP: fragment {frag_smi} already present in query "
#                       f"at/near atom {query_attach_atom}")
#                 return True
#         return False

#     simple_tasks = [
#         t for t in simple_tasks
#         if not _fragment_already_present(
#             t["frag_atoms_list"], t["query_attach_atom"]
#         )
#     ]

#     fused_ring_tasks = [
#         (a, b) for a, b in fused_ring_tasks
#         if not _fragment_already_present(
#             a["frag_atoms_list"], a["query_attach_atom"]
#         )
#     ]
#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------
#     def _add_fragment_atoms(frag_atoms_list):
#         ref_to_new = {}
#         for ref_atom_idx in frag_atoms_list:
#             ref_atom = ref_mol.GetAtomWithIdx(ref_atom_idx)
#             new_atom = Chem.Atom(ref_atom.GetAtomicNum())
#             new_atom.SetFormalCharge(ref_atom.GetFormalCharge())
#             new_atom.SetNumExplicitHs(ref_atom.GetNumExplicitHs())
#             new_atom.SetIsAromatic(False)
#             new_idx = new_mol.AddAtom(new_atom)
#             ref_to_new[ref_atom_idx] = new_idx

#         # ── Kekulize ref_mol once so aromatic bonds have explicit SINGLE/DOUBLE ──
#         ref_mol_kek = Chem.RWMol(ref_mol)
#         try:
#             Chem.Kekulize(ref_mol_kek, clearAromaticFlags=False)  # in-place on copy
#         except Exception:
#             pass  # if it fails, fall back to original bond types below

#         frag_set = set(frag_atoms_list)
#         for ref_atom_idx in frag_atoms_list:
#             ref_atom = ref_mol_kek.GetAtomWithIdx(ref_atom_idx)
#             for nb in ref_atom.GetNeighbors():
#                 nb_idx = nb.GetIdx()
#                 if nb_idx not in frag_set or nb_idx <= ref_atom_idx:
#                     continue
#                 bond  = ref_mol_kek.GetBondBetweenAtoms(ref_atom_idx, nb_idx)
#                 btype = bond.GetBondType()
#                 # After Kekulize, AROMATIC bonds become SINGLE or DOUBLE —
#                 # but if Kekulize failed and bond is still AROMATIC, fall back to SINGLE
#                 if btype == Chem.BondType.AROMATIC:
#                     btype = Chem.BondType.SINGLE
#                 new_mol.AddBond(ref_to_new[ref_atom_idx], ref_to_new[nb_idx], btype)

#         return ref_to_new

#     def _find_attach(query_attach_atom, ref_boundary_coord):
#         return find_attachment_point(
#             new_mol                 = new_mol,
#             query_attach_atom       = query_attach_atom,
#             query_mol_original      = query_mol,
#             query_coords_aligned    = query_coords_aligned,
#             ref_boundary_atom_coord = ref_boundary_coord,
#         )

#     def _remove_h_if_needed(attach_idx, needs_h):
#         if needs_h:
#             a = new_mol.GetAtomWithIdx(attach_idx)
#             a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - 1))

#     def _add_anchor_bond(attach_idx, new_root_idx, bond_type):
#         bt = bond_type
#         if bt == Chem.BondType.AROMATIC:
#             bt = Chem.BondType.SINGLE
#         new_mol.AddBond(attach_idx, new_root_idx, bt)

#     fragment_map = {}   # attach_idx → fragment SMILES (for reporting)
#     # ------------------------------------------------------------------
#     # Step 3a: simple (single-anchor) fragments
#     # ------------------------------------------------------------------
#     for task in simple_tasks:
#         ref_boundary_atom  = task["ref_boundary_atom"]
#         query_attach_atom  = task["query_attach_atom"]
#         frag_atoms         = task["frag_atoms_list"]
#         root_nb            = task["root_nb"]
#         bond_type          = task["bond_type"]
#         ref_boundary_coord = get_ref_atom_coord(ref_boundary_atom)

#         attach_idx, needs_h = _find_attach(query_attach_atom, ref_boundary_coord)
#         if attach_idx is None:
#             failed_log.append({
#                 "ref_boundary_atom": ref_boundary_atom,
#                 "query_attach_atom": query_attach_atom,
#                 "reason": "no free valence within 2 hops",
#                 "fragment_atoms": frag_atoms,
#             })
#             print(f"  FAILED: no attachment point from ref atom {ref_boundary_atom}")
#             continue

#         _remove_h_if_needed(attach_idx, needs_h)
#         ref_to_new = _add_fragment_atoms(frag_atoms)
#         _add_anchor_bond(attach_idx, ref_to_new[root_nb], bond_type)

#         attachment_log.append({
#             "ref_boundary_atom":  ref_boundary_atom,
#             "query_attach_atom":  query_attach_atom,
#             "actual_attach_atom": attach_idx,
#             "redirected":         attach_idx != query_attach_atom,
#             "fragment_ref_atoms": frag_atoms,
#             "n_atoms_added":      len(frag_atoms),
#             "was_fused_ring":     False,
#         })
#         print(f"  Attached (simple): ref atoms {frag_atoms} → query atom {attach_idx}"
#               + (" [redirected]" if attach_idx != query_attach_atom else ""))

#         fsmi = _frag_smiles(frag_atoms)
#         if fsmi:
#             fragment_map[attach_idx] = fsmi
#     # ------------------------------------------------------------------
#     # Step 3b: fused-ring fragments (two anchors, one fragment)
#     # ------------------------------------------------------------------
#     for task_a, task_b in fused_ring_tasks:
#         frag_atoms = task_a["frag_atoms_list"]

#         ref_coord_a = get_ref_atom_coord(task_a["ref_boundary_atom"])
#         ref_coord_b = get_ref_atom_coord(task_b["ref_boundary_atom"])

#         attach_a, needs_h_a = _find_attach(task_a["query_attach_atom"], ref_coord_a)
#         attach_b, needs_h_b = _find_attach(task_b["query_attach_atom"], ref_coord_b)

#         if attach_a is None or attach_b is None:
#             failed_log.append({
#                 "ref_boundary_atoms": (task_a["ref_boundary_atom"],
#                                        task_b["ref_boundary_atom"]),
#                 "reason": "fused ring: no free valence at one or both anchor points",
#                 "fragment_atoms": frag_atoms,
#             })
#             print(f"  FAILED: fused ring — attach_a={attach_a}, attach_b={attach_b}")
#             continue

#         _remove_h_if_needed(attach_a, needs_h_a)
#         _remove_h_if_needed(attach_b, needs_h_b)

#         ref_to_new = _add_fragment_atoms(frag_atoms)

#         # First anchor: query_attach_a → root of task_a (one end of chain)
#         _add_anchor_bond(attach_a, ref_to_new[task_a["root_nb"]], task_a["bond_type"])

#         # Second anchor: query_attach_b → root of task_b (closes the ring)
#         new_root_b = ref_to_new[task_b["root_nb"]]
#         if new_mol.GetBondBetweenAtoms(attach_b, new_root_b) is None:
#             _add_anchor_bond(attach_b, new_root_b, task_b["bond_type"])
#         else:
#             print(f"  SKIP duplicate ring-closing bond {attach_b}↔{new_root_b}")

#         print(f"  Attached (fused ring): ref atoms {frag_atoms}")
#         print(f"    Anchor A: query atom {attach_a} ↔ frag root ref {task_a['root_nb']}"
#               + (" [redirected]" if attach_a != task_a["query_attach_atom"] else ""))
#         print(f"    Anchor B: query atom {attach_b} ↔ frag root ref {task_b['root_nb']}"
#               + (" [redirected]" if attach_b != task_b["query_attach_atom"] else ""))

#         attachment_log.append({
#             "ref_boundary_atoms":  (task_a["ref_boundary_atom"],
#                                     task_b["ref_boundary_atom"]),
#             "query_attach_atoms":  (task_a["query_attach_atom"],
#                                     task_b["query_attach_atom"]),
#             "actual_attach_atoms": (attach_a, attach_b),
#             "fragment_ref_atoms":  frag_atoms,
#             "n_atoms_added":       len(frag_atoms),
#             "was_fused_ring":      True,
#         })

#         fsmi = _frag_smiles(frag_atoms)
#         if fsmi:
#             fragment_map[attach_a] = fsmi   # primary anchor carries the label
#             fragment_map[attach_b] = fsmi   # both anchors point to same ring

#     # ------------------------------------------------------------------
#     # Step 4: selectively fix only the NEW bonds that connect query atoms
#     # to fragment atoms — these are the junction bonds that may carry
#     # stale AROMATIC flags from the original query mol.
#     #
#     # We do NOT touch the original query mol atoms/bonds at all.
#     # SanitizeMol will re-perceive aromaticity for the whole mol,
#     # including the original aromatic rings, correctly.
#     # ------------------------------------------------------------------
#     newly_added_indices = set(range(original_atom_count, new_mol.GetNumAtoms()))

#     for bond in new_mol.GetBonds():
#         i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         # Only fix bonds where at least one end is a newly added atom
#         if i in newly_added_indices or j in newly_added_indices:
#             if bond.GetBondType() == Chem.BondType.AROMATIC:
#                 bond.SetBondType(Chem.BondType.SINGLE)
#             bond.SetIsAromatic(False)

#     # Also clear IsAromatic on newly added atoms (they were set False at
#     # creation but be defensive in case RDKit set them during AddBond)
#     for idx in newly_added_indices:
#         new_mol.GetAtomWithIdx(idx).SetIsAromatic(False)

#     # ------------------------------------------------------------------
#     # Step 5: sanitize — original query aromaticity is untouched,
#     # new fragment atoms get aromaticity re-perceived from scratch
#     # ------------------------------------------------------------------
#     try:
#         Chem.SanitizeMol(new_mol)
#         mol = new_mol.GetMol()
#         Chem.Kekulize(mol, clearAromaticFlags=False)
#         print(f"\n  Sanitization OK")
#         mol_smiles = Chem.MolToSmiles(mol)
#         print(f"  Result SMILES: {mol_smiles}")
#         return mol_smiles, attachment_log, failed_log, fragment_map

#     except Chem.KekulizeException as e:
#         print(f"\n  Kekulization failed: {e}")

#     except Exception as e:
#         print(f"\n  Sanitization failed: {e}")

#     # Final fallback
#     print("  Attempting SMILES round-trip...")
#     try:
#         raw_smi = Chem.MolToSmiles(new_mol.GetMol(), allHsExplicit=False)
#         print(f"  Raw SMILES: {raw_smi}")
#         mol = Chem.MolFromSmiles(raw_smi)
#         if mol:
#             print(f"  Round-trip OK: {Chem.MolToSmiles(mol)}")
#             return raw_smi, attachment_log, failed_log, fragment_map
#         print("  Round-trip returned None")
#     except Exception as e:
#         print(f"  Round-trip failed: {e}")

#     return None, attachment_log, failed_log,fragment_map

def attach_fragments_to_query(
    query_mol,
    ref_mol,
    boundary_to_query_map,
    replaceable_set,
    ref_to_query_atom_map,
    query_coords_aligned,
):
    from rdkit.Chem import RWMol
    from collections import defaultdict

    boundary_to_query_map = {int(k): int(v) for k, v in boundary_to_query_map.items()}
    replaceable_set       = set(int(a) for a in replaceable_set)
    ref_to_query_atom_map = {int(k): int(v) for k, v in ref_to_query_atom_map.items()}
    query_coords_aligned  = {int(k): v for k, v in query_coords_aligned.items()}

    attachment_log = []
    failed_log     = []

    new_mol  = RWMol(Chem.RWMol(query_mol))
    ref_conf = ref_mol.GetConformer()

    # Track which atom indices in new_mol were freshly added (not from query_mol)
    original_atom_count = new_mol.GetNumAtoms()

    def get_ref_atom_coord(atom_idx):
        pos = ref_conf.GetAtomPosition(int(atom_idx))
        return np.array([pos.x, pos.y])

     # ── NEW helper ──────────────────────────────────────────────────────────
    def _frag_smiles(frag_atoms_list):
        """SMILES of the ref fragment atoms (no attachment marker)."""
        try:
            smi = Chem.MolFragmentToSmiles(ref_mol, frag_atoms_list)
            return smi if smi else None
        except Exception:
            return None
    # ────────────────────────────────────────────────────────────────────────
    # ------------------------------------------------------------------
    # Step 1: collect all tasks before doing anything
    # ------------------------------------------------------------------
    all_tasks = []

    for ref_boundary_atom, query_attach_atom in boundary_to_query_map.items():
        ref_boundary_atom = int(ref_boundary_atom)
        query_attach_atom = int(query_attach_atom)

        all_fragment_atoms, root_neighbors, bond_types, is_ring_fragment = \
            extract_fragment(ref_mol, ref_boundary_atom, replaceable_set)

        if not all_fragment_atoms:
            failed_log.append({
                "ref_boundary_atom": ref_boundary_atom,
                "query_attach_atom": query_attach_atom,
                "reason": "no fragment outside replaceable region",
            })
            continue

        for frag_info, root_nb, bond_type in zip(
                all_fragment_atoms, root_neighbors, bond_types):
            if not frag_info["atoms"]:
                continue
            all_tasks.append({
                "ref_boundary_atom": ref_boundary_atom,
                "query_attach_atom": query_attach_atom,
                "frag_key":          tuple(sorted(frag_info["atoms"])),
                "frag_atoms_list":   frag_info["atoms"],
                "is_partial_ring":   frag_info["is_partial_ring"],
                "root_nb":           root_nb,
                "bond_type":         bond_type,
            })


    # ------------------------------------------------------------------
    # Step 2: group by fragment atom set → detect fused rings
    # ------------------------------------------------------------------
    frag_key_to_tasks = defaultdict(list)
    for task in all_tasks:
        frag_key_to_tasks[task["frag_key"]].append(task)

    simple_tasks     = []
    fused_ring_tasks = []

    for frag_key, tasks in frag_key_to_tasks.items():
        partial = [t for t in tasks if t["is_partial_ring"]]
        normal  = [t for t in tasks if not t["is_partial_ring"]]

        for t in normal:
            simple_tasks.append(t)

        if len(partial) == 2:
            fused_ring_tasks.append((partial[0], partial[1]))
            print(f"  FUSED RING DETECTED: fragment {frag_key} shared by "
                  f"ref boundary atoms {partial[0]['ref_boundary_atom']} and "
                  f"{partial[1]['ref_boundary_atom']} → "
                  f"query atoms {partial[0]['query_attach_atom']} and "
                  f"{partial[1]['query_attach_atom']}")
        elif len(partial) == 1:
            simple_tasks.append(partial[0])
        elif len(partial) > 2:
            fused_ring_tasks.append((partial[0], partial[1]))
            for t in partial[2:]:
                simple_tasks.append(t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add_fragment_atoms(frag_atoms_list):
        ref_to_new = {}
        for ref_atom_idx in frag_atoms_list:
            ref_atom = ref_mol.GetAtomWithIdx(ref_atom_idx)
            new_atom = Chem.Atom(ref_atom.GetAtomicNum())
            new_atom.SetFormalCharge(ref_atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(ref_atom.GetNumExplicitHs())
            new_atom.SetIsAromatic(False)
            new_idx = new_mol.AddAtom(new_atom)
            ref_to_new[ref_atom_idx] = new_idx

        # ── Kekulize ref_mol once so aromatic bonds have explicit SINGLE/DOUBLE ──
        ref_mol_kek = Chem.RWMol(ref_mol)
        try:
            Chem.Kekulize(ref_mol_kek, clearAromaticFlags=False)  # in-place on copy
        except Exception:
            pass  # if it fails, fall back to original bond types below

        frag_set = set(frag_atoms_list)
        for ref_atom_idx in frag_atoms_list:
            ref_atom = ref_mol_kek.GetAtomWithIdx(ref_atom_idx)
            for nb in ref_atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx not in frag_set or nb_idx <= ref_atom_idx:
                    continue
                bond  = ref_mol_kek.GetBondBetweenAtoms(ref_atom_idx, nb_idx)
                btype = bond.GetBondType()
                # After Kekulize, AROMATIC bonds become SINGLE or DOUBLE —
                # but if Kekulize failed and bond is still AROMATIC, fall back to SINGLE
                if btype == Chem.BondType.AROMATIC:
                    btype = Chem.BondType.SINGLE
                new_mol.AddBond(ref_to_new[ref_atom_idx], ref_to_new[nb_idx], btype)

        return ref_to_new

    def _find_attach(query_attach_atom, ref_boundary_coord):
        return find_attachment_point(
            new_mol                 = new_mol,
            query_attach_atom       = query_attach_atom,
            query_mol_original      = query_mol,
            query_coords_aligned    = query_coords_aligned,
            ref_boundary_atom_coord = ref_boundary_coord,
        )

    def _remove_h_if_needed(attach_idx, needs_h):
        if needs_h:
            a = new_mol.GetAtomWithIdx(attach_idx)
            a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - 1))

    def _add_anchor_bond(attach_idx, new_root_idx, bond_type):
        bt = bond_type
        if bt == Chem.BondType.AROMATIC:
            bt = Chem.BondType.SINGLE
        new_mol.AddBond(attach_idx, new_root_idx, bt)

    fragment_map = {}   # attach_idx → fragment SMILES (for reporting)
    # ------------------------------------------------------------------
    # Step 3a: simple (single-anchor) fragments
    # ------------------------------------------------------------------
    for task in simple_tasks:
        ref_boundary_atom  = task["ref_boundary_atom"]
        query_attach_atom  = task["query_attach_atom"]
        frag_atoms         = task["frag_atoms_list"]
        root_nb            = task["root_nb"]
        bond_type          = task["bond_type"]
        ref_boundary_coord = get_ref_atom_coord(ref_boundary_atom)

        attach_idx, needs_h = _find_attach(query_attach_atom, ref_boundary_coord)
        if attach_idx is None:
            failed_log.append({
                "ref_boundary_atom": ref_boundary_atom,
                "query_attach_atom": query_attach_atom,
                "reason": "no free valence within 2 hops",
                "fragment_atoms": frag_atoms,
            })
            print(f"  FAILED: no attachment point from ref atom {ref_boundary_atom}")
            continue

        _remove_h_if_needed(attach_idx, needs_h)
        ref_to_new = _add_fragment_atoms(frag_atoms)
        _add_anchor_bond(attach_idx, ref_to_new[root_nb], bond_type)

        attachment_log.append({
            "ref_boundary_atom":  ref_boundary_atom,
            "query_attach_atom":  query_attach_atom,
            "actual_attach_atom": attach_idx,
            "redirected":         attach_idx != query_attach_atom,
            "fragment_ref_atoms": frag_atoms,
            "n_atoms_added":      len(frag_atoms),
            "was_fused_ring":     False,
        })
        print(f"  Attached (simple): ref atoms {frag_atoms} → query atom {attach_idx}"
              + (" [redirected]" if attach_idx != query_attach_atom else ""))

        fsmi = _frag_smiles(frag_atoms)
        if fsmi:
            fragment_map[attach_idx] = fsmi
    # ------------------------------------------------------------------
    # Step 3b: fused-ring fragments (two anchors, one fragment)
    # ------------------------------------------------------------------
    for task_a, task_b in fused_ring_tasks:
        frag_atoms = task_a["frag_atoms_list"]

        ref_coord_a = get_ref_atom_coord(task_a["ref_boundary_atom"])
        ref_coord_b = get_ref_atom_coord(task_b["ref_boundary_atom"])

        attach_a, needs_h_a = _find_attach(task_a["query_attach_atom"], ref_coord_a)
        attach_b, needs_h_b = _find_attach(task_b["query_attach_atom"], ref_coord_b)

        if attach_a is None or attach_b is None:
            failed_log.append({
                "ref_boundary_atoms": (task_a["ref_boundary_atom"],
                                       task_b["ref_boundary_atom"]),
                "reason": "fused ring: no free valence at one or both anchor points",
                "fragment_atoms": frag_atoms,
            })
            print(f"  FAILED: fused ring — attach_a={attach_a}, attach_b={attach_b}")
            continue

        # ── NEW: if both anchors resolved to the same query atom,
        # it's not a genuine ring closure — treat as simple attachment
        if attach_a == attach_b:
            print(f"  FUSED RING COLLAPSED: both anchors resolved to same "
                  f"query atom {attach_a} — treating as simple attachment from task_a only")
            _remove_h_if_needed(attach_a, needs_h_a)
            ref_to_new = _add_fragment_atoms(frag_atoms)
            _add_anchor_bond(attach_a, ref_to_new[task_a["root_nb"]], task_a["bond_type"])
            attachment_log.append({
                "ref_boundary_atoms":  (task_a["ref_boundary_atom"],
                                        task_b["ref_boundary_atom"]),
                "query_attach_atoms":  (task_a["query_attach_atom"],
                                        task_b["query_attach_atom"]),
                "actual_attach_atoms": (attach_a,),
                "fragment_ref_atoms":  frag_atoms,
                "n_atoms_added":       len(frag_atoms),
                "was_fused_ring":      False,
                "collapsed":           True,
            })
            fsmi = _frag_smiles(frag_atoms)
            if fsmi:
                fragment_map[attach_a] = fsmi
            continue   # ← skip the rest of the fused ring block

        _remove_h_if_needed(attach_a, needs_h_a)
        _remove_h_if_needed(attach_b, needs_h_b)

        ref_to_new = _add_fragment_atoms(frag_atoms)

        # First anchor: query_attach_a → root of task_a (one end of chain)
        _add_anchor_bond(attach_a, ref_to_new[task_a["root_nb"]], task_a["bond_type"])

        # Second anchor: query_attach_b → root of task_b (closes the ring)
        new_root_b = ref_to_new[task_b["root_nb"]]
        if new_mol.GetBondBetweenAtoms(attach_b, new_root_b) is None:
            _add_anchor_bond(attach_b, new_root_b, task_b["bond_type"])
        else:
            print(f"  SKIP duplicate ring-closing bond {attach_b}↔{new_root_b}")

        print(f"  Attached (fused ring): ref atoms {frag_atoms}")
        print(f"    Anchor A: query atom {attach_a} ↔ frag root ref {task_a['root_nb']}"
              + (" [redirected]" if attach_a != task_a["query_attach_atom"] else ""))
        print(f"    Anchor B: query atom {attach_b} ↔ frag root ref {task_b['root_nb']}"
              + (" [redirected]" if attach_b != task_b["query_attach_atom"] else ""))

        attachment_log.append({
            "ref_boundary_atoms":  (task_a["ref_boundary_atom"],
                                    task_b["ref_boundary_atom"]),
            "query_attach_atoms":  (task_a["query_attach_atom"],
                                    task_b["query_attach_atom"]),
            "actual_attach_atoms": (attach_a, attach_b),
            "fragment_ref_atoms":  frag_atoms,
            "n_atoms_added":       len(frag_atoms),
            "was_fused_ring":      True,
        })

        fsmi = _frag_smiles(frag_atoms)
        if fsmi:
            fragment_map[attach_a] = fsmi   # primary anchor carries the label
            fragment_map[attach_b] = fsmi   # both anchors point to same ring

    # ------------------------------------------------------------------
    # Step 4: selectively fix only the NEW bonds that connect query atoms
    # to fragment atoms — these are the junction bonds that may carry
    # stale AROMATIC flags from the original query mol.
    #
    # We do NOT touch the original query mol atoms/bonds at all.
    # SanitizeMol will re-perceive aromaticity for the whole mol,
    # including the original aromatic rings, correctly.
    # ------------------------------------------------------------------
    newly_added_indices = set(range(original_atom_count, new_mol.GetNumAtoms()))

    for bond in new_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Only fix bonds where at least one end is a newly added atom
        if i in newly_added_indices or j in newly_added_indices:
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)

    # Also clear IsAromatic on newly added atoms (they were set False at
    # creation but be defensive in case RDKit set them during AddBond)
    for idx in newly_added_indices:
        new_mol.GetAtomWithIdx(idx).SetIsAromatic(False)

    # ------------------------------------------------------------------
    # Step 5: sanitize — original query aromaticity is untouched,
    # new fragment atoms get aromaticity re-perceived from scratch
    # ------------------------------------------------------------------
    try:
        Chem.SanitizeMol(new_mol)
        mol = new_mol.GetMol()
        Chem.Kekulize(mol, clearAromaticFlags=False)
        print(f"\n  Sanitization OK")
        mol_smiles = Chem.MolToSmiles(mol)
        print(f"  Result SMILES: {mol_smiles}")
        return mol_smiles, attachment_log, failed_log, fragment_map

    except Chem.KekulizeException as e:
        print(f"\n  Kekulization failed: {e}")

    except Exception as e:
        print(f"\n  Sanitization failed: {e}")

    # Final fallback
    print("  Attempting SMILES round-trip...")
    try:
        raw_smi = Chem.MolToSmiles(new_mol.GetMol(), allHsExplicit=False)
        print(f"  Raw SMILES: {raw_smi}")
        mol = Chem.MolFromSmiles(raw_smi)
        if mol:
            print(f"  Round-trip OK: {Chem.MolToSmiles(mol)}")
            return raw_smi, attachment_log, failed_log, fragment_map
        print("  Round-trip returned None")
    except Exception as e:
        print(f"  Round-trip failed: {e}")

    return None, attachment_log, failed_log,fragment_map

# 
# =============================================================================
# Orchestrator for scaffold hopping
# =============================================================================

def build_hopped_scaffold(
    align_result,
    region_info,
    ref_mol,
    query_mol,
    query_coords_aligned,        # ← add this
):
    """
    Full fragment transplantation pipeline.

    Parameters
    ----------
    align_result          : (R, t, atom_mapping, rmsd, region_info)
    region_info           : region_info dict from align_and_visualize
    ref_mol               : RDKit mol with conformer
    query_mol             : RDKit mol with conformer
    query_coords_aligned  : {atom_idx: aligned_2d_coords} from align step
    """
    R, t, atom_mapping, rmsd, region_info = align_result   # ← 5 values, correct
    replaceable_set       = set(region_info["replaceable_atom_indices"])
    boundary_to_query_map = region_info["boundary_to_query_map"]

    # Build flat ref→query atom map from deduplicated atom_mapping
    ref_to_query_atom_map = {}
    for entry in atom_mapping:
        for r_idx, q_idx, dist in entry["atom_pairs"]:
            ref_to_query_atom_map[r_idx] = q_idx

    print("\n" + "=" * 60)
    print("FRAGMENT TRANSPLANTATION")
    print("=" * 60)
    print(f"  Replaceable region : {sorted(replaceable_set)}")
    print(f"  Boundary atoms     : {list(boundary_to_query_map.keys())}")
    print(f"  Boundary→Query     : {boundary_to_query_map}")
    print()

    hopped_mol_smiles, attachment_log, failed_log, fragment_map = attach_fragments_to_query(
        query_mol              = query_mol,
        ref_mol                = ref_mol,
        boundary_to_query_map  = boundary_to_query_map,
        replaceable_set        = replaceable_set,
        ref_to_query_atom_map  = ref_to_query_atom_map,
        query_coords_aligned   = query_coords_aligned,   # ← pass through
    )

    print(f"\n  Attachments succeeded : {len(attachment_log)}")
    print(f"  Attachments failed    : {len(failed_log)}")

    if failed_log:
        print("\n  Failed attachments:")
        for f in failed_log:
            print(f"    ref atom {f['ref_boundary_atom']} → "
                  f"query atom {f['query_attach_atom']} : {f['reason']}")

    return hopped_mol_smiles, attachment_log, failed_log, fragment_map


# ==============================================================================================================================
#                                                  Molecule Generation -- FARE
# =============================================================================================================================


# =============================================================================
# Molecular Descriptors
# =============================================================================
def calc_props(smi):
    """Calculate basic molecular descriptors for a given SMILES.

    Parameters
    ----------
    smi : str
        SMILES string of the molecule

    Returns
    -------
    dict
        Dictionary containing: MolWt, LogP, TPSA, HBA, HBD, RotatableBonds
    """
    if smi is None:
        
        return None

    smi = str(smi).strip()

    if smi == "":
        return None

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol)
    }


# =============================================================================
# Fragment Processing
# =============================================================================
def process_fragments(fragment_df, calc_props_func):
    """Process fragment library and calculate descriptors."""
    pattern = r"\[[^\]]*\*[^\]]*\]"
    fragment_df = fragment_df.copy()
    fragment_df["att_repl"] = fragment_df["Fragments"].str.replace(pattern, "C", regex=True)

    desc_dict = []
    for idx, smi in fragment_df["att_repl"].items():
        try:
            if pd.isna(smi) or smi is None:
                raise ValueError("Null SMILES")
            props = calc_props_func(smi)
            if props is None:
                raise ValueError("calc_props_func returned None")
            desc_dict.append({"_idx": idx, **props})
        except Exception:
            desc_dict.append({"_idx": idx})

    desc_df = pd.DataFrame(desc_dict).set_index("_idx")
    fragment_df = fragment_df.join(desc_df, how="left")
    return fragment_df


def process_and_save_similar_fragments(frag_repl, fragment_desc, cutoff, save_dir="Similar_frag"):
    """Find similar fragments based on descriptor similarity."""
    os.makedirs(save_dir, exist_ok=True)
    similar_dfs = {}
    fragment_desc = fragment_desc.drop_duplicates(subset=["att_repl"])

    for att_point, frag_smi in frag_repl.items():
        frag_props = calc_props(frag_smi)
        if frag_props is None:
            continue

        similarities = []
        for _, row in fragment_desc.iterrows():
            sim = 0
            for prop in ['MolWt', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotatableBonds']:
                sim += 1 - abs(row[prop] - frag_props[prop]) / (abs(row[prop]) + 1e-6)
            sim /= 6
            similarities.append(sim)

        fragment_desc["Similarity"] = similarities
        filtered_df = fragment_desc.nlargest(cutoff, "Similarity").copy().reset_index(drop=True)
        filtered_df = filtered_df.dropna(subset=['Fragments'])

        similar_dfs[att_point] = filtered_df
        csv_path = os.path.join(save_dir, f"att_{att_point}.csv")
        filtered_df.to_csv(csv_path, index=False)

    return similar_dfs


def att_to_c_conversion(mol, keep_idx, att_indices):
    """Convert specified atoms to carbon, except the designated keep_idx."""
    rw = Chem.RWMol(mol)
    for idx in att_indices:
        if idx != keep_idx:
            rw.GetAtomWithIdx(idx).SetAtomicNum(6)
    return rw.GetMol()


def prepare_fragments(fragment_df):
    """Process fragment DataFrame to generate molecules with attachment points."""
    fragment_df = fragment_df.dropna(subset=["Fragments"]).reset_index(drop=True)
    fragment_df["mol"] = fragment_df["Fragments"].apply(lambda x: Chem.MolFromSmiles(x))
    fragment_df['dupli_count'] = fragment_df['mol'].apply(
        lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "*")
    )
    fragment_df = fragment_df[fragment_df['dupli_count'] > 0].reset_index(drop=True)

    fragment_df['att_points'] = fragment_df['mol'].apply(
        lambda mol: [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
    )

    duplicated = []
    for i, m in enumerate(fragment_df['dupli_count']):
        row = fragment_df.loc[[i]]
        duplicated.append(pd.concat([row] * m, ignore_index=True))
    expanded = pd.concat(duplicated, ignore_index=True)

    new_mols, frag_atts = [], []
    for i in range(len(expanded)):
        mol = expanded.loc[i, 'mol']
        att_indices = expanded.loc[i, 'att_points']
        dupli_idx = expanded.loc[i, 'dupli_count']
        keep_idx = att_indices[i % dupli_idx]
        new_mols.append(att_to_c_conversion(mol, keep_idx, att_indices))
        frag_atts.append(keep_idx)

    expanded['new_mol'] = new_mols
    expanded['frag_att'] = frag_atts
    return expanded


# =============================================================================
# Molecule Generation
# =============================================================================
def remove_dummy_atoms(mol):
    """Remove all dummy atoms (*) from a molecule."""
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if not dummy_indices:
        return mol

    edit = Chem.RWMol(mol)
    for idx in sorted(dummy_indices, reverse=True):
        edit.RemoveAtom(idx)

    cleaned = edit.GetMol()
    Chem.SanitizeMol(cleaned)
    return cleaned


def generate_final_molecules(combinations, scaffold_mol, similar_dfs, att_to_c_conversion):
    """Generate combined molecules from scaffold and fragments."""

    # ── valence helper (from attach_fragments_to_query) ──────────────────────
    def has_free_valence(mol, idx):
        atom = mol.GetAtomWithIdx(idx)
        try:
            atom.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        try:
            if atom.GetNumImplicitHs() > 0:
                return True, False
        except RuntimeError:
            pass
        if atom.GetNumExplicitHs() > 0:
            return True, True   # needs explicit H removal
        return False, False

    def remove_h_if_needed(rwmol, idx):
        _, needs_h = has_free_valence(rwmol, idx)
        if needs_h:
            a = rwmol.GetAtomWithIdx(idx)
            a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - 1))
    # ─────────────────────────────────────────────────────────────────────────

    final_mols = []

    for c in combinations:
        try:
            mol_num_i = scaffold_mol.GetNumAtoms()

            for i, mol_i in enumerate(c):
                att_pt = list(similar_dfs.keys())[i]
                frag_i = similar_dfs[att_pt].loc[mol_i, 'new_mol']
                frag_i = att_to_c_conversion(
                    frag_i, None,
                    [int(similar_dfs[att_pt].loc[mol_i, "frag_att"])]
                )

                if i == 0:
                    frag_att = mol_num_i + similar_dfs[att_pt].loc[mol_i, "frag_att"]
                    combo = Chem.CombineMols(scaffold_mol, frag_i)
                    combo_rw = Chem.RWMol(combo)                   # ← RWMol for H removal
                    remove_h_if_needed(combo_rw, int(att_pt))      # ← scaffold side
                    remove_h_if_needed(combo_rw, int(frag_att))    # ← fragment side
                    combo_editable = Chem.EditableMol(combo_rw)
                    combo_editable.AddBond(
                        int(att_pt), int(frag_att),
                        order=Chem.rdchem.BondType.SINGLE
                    )
                else:
                    current_mol = combo_editable.GetMol()
                    new_mol_num_i = current_mol.GetNumAtoms()
                    frag_att = new_mol_num_i + similar_dfs[att_pt].loc[mol_i, "frag_att"]
                    combo = Chem.CombineMols(current_mol, frag_i)
                    combo_rw = Chem.RWMol(combo)                   # ← RWMol for H removal
                    remove_h_if_needed(combo_rw, int(att_pt))      # ← scaffold side
                    remove_h_if_needed(combo_rw, int(frag_att))    # ← fragment side
                    combo_editable = Chem.EditableMol(combo_rw)
                    combo_editable.AddBond(
                        int(att_pt), int(frag_att),
                        order=Chem.rdchem.BondType.SINGLE
                    )

            combined_mol = combo_editable.GetMol()
            combined_mol = remove_dummy_atoms(combined_mol)
            Chem.SanitizeMol(combined_mol)
            final_mols.append(combined_mol)

        except Exception:
            continue

    for mol in final_mols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    smi = [Chem.MolToSmiles(mol) for mol in final_mols]

    def clean_bracketed_carbons(smiles):
        fixed = re.sub(r"\[\d*C\]", "C", smiles)
        mol = Chem.MolFromSmiles(fixed)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    cleaned_smi = [clean_bracketed_carbons(x) for x in smi]
    return pd.DataFrame({"smiles": cleaned_smi})


# =============================================================================
# BRICS Fragmentation
# =============================================================================
def brics_split(mol, target_set):
    """Split molecule using BRICS decomposition targeting specific atoms."""
    n = mol.GetNumAtoms()
    brics_bonds = list(BRICS.FindBRICSBonds(mol))

    if not brics_bonds:
        return frozenset(range(n)), frozenset(), [], mol, mol

    bond_idx = [mol.GetBondBetweenAtoms(a, b).GetIdx() for (a, b), _ in brics_bonds]
    frag_mol = Chem.FragmentOnBonds(mol, bond_idx, addDummies=True)
    grp_tuples = Chem.GetMolFrags(frag_mol)
    grp_mols = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

    frags_orig = [frozenset(i for i in g if i < n) for g in grp_tuples]
    target_set = frozenset(target_set)
    all_orig = frozenset(range(n))

    def score(atom_set):
        ov = len(target_set & atom_set)
        return (ov, ov / len(atom_set) if atom_set else 0.0)

    def count_attachment_bonds(frag_atoms):
        return sum(1 for (a, b), _ in brics_bonds if (a in frag_atoms) != (b in frag_atoms))

    best_single_i = max(range(len(frags_orig)), key=lambda i: score(frags_orig[i]))
    best_single = frags_orig[best_single_i]
    score_single = score(best_single)

    best_complement = None
    best_complement_excl = None
    score_complement = (-1, -1.0)

    for i, frag in enumerate(frags_orig):
        n_attach = count_attachment_bonds(frag)
        if n_attach != 1:
            continue

        complement = all_orig - frag
        s = score(complement)
        if s > score_complement:
            score_complement = s
            best_complement = complement
            best_complement_excl = frag

    use_complement = (best_complement is not None) and (score_complement > score_single)

    if use_complement:
        chosen_atoms = best_complement
        removed_atoms = best_complement_excl
    else:
        chosen_atoms = best_single
        removed_atoms = all_orig - best_single

    cut_bonds = [(a, b) for (a, b), _ in brics_bonds if (a in chosen_atoms) != (b in chosen_atoms)]

    boundary_bond_indices = [mol.GetBondBetweenAtoms(a, b).GetIdx() for (a, b) in cut_bonds]

    if not boundary_bond_indices:
        return chosen_atoms, removed_atoms, cut_bonds, mol, mol

    cut_mol = Chem.FragmentOnBonds(mol, boundary_bond_indices, addDummies=True)
    cut_tuples = Chem.GetMolFrags(cut_mol)
    cut_mols = Chem.GetMolFrags(cut_mol, asMols=True, sanitizeFrags=False)
    cut_frags_orig = [frozenset(i for i in g if i < n) for g in cut_tuples]

    if use_complement:
        chosen_i = max(range(len(cut_frags_orig)), key=lambda i: len(chosen_atoms & cut_frags_orig[i]))
        frag_with_dummy = cut_mols[chosen_i]

        other_indices = [i for i in range(len(cut_frags_orig)) if i != chosen_i]
        core_frag = cut_mols[other_indices[0]]
        if len(other_indices) > 1:
            for oi in other_indices[1:]:
                core_frag = Chem.CombineMols(core_frag, cut_mols[oi])
    else:
        chosen_i = max(range(len(cut_frags_orig)), key=lambda i: len(chosen_atoms & cut_frags_orig[i]))
        frag_with_dummy = cut_mols[chosen_i]

        other_indices = [i for i in range(len(cut_frags_orig)) if i != chosen_i]
        if len(other_indices) == 1:
            core_frag = cut_mols[other_indices[0]]
        else:
            core_frag = cut_mols[other_indices[0]]
            for oi in other_indices[1:]:
                core_frag = Chem.CombineMols(core_frag, cut_mols[oi])

    return chosen_atoms, removed_atoms, cut_bonds, frag_with_dummy, core_frag


# =============================================================================
# Visualization of Scaffold Hopping Progression
# =============================================================================

from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def draw_scaffold_hop(
    ref_smiles: str,
    scaffold_smiles: str,
    hopped_smiles: str,
    ref_label: str = "Reference",
    scaffold_label: str = "Scaffold",
    hopped_label: str = "Hopped Molecule",
    highlight_color: tuple = (1.0, 0.7, 0.4),
    mol_size: tuple = (450, 350),
    arrow_space: int = 90,
    save_path: str = "scaffold_hopping_progression.png",
    show: bool = True,
) -> Image.Image:
    """
    Generate a scaffold hopping progression image showing:
        Reference Molecule → Scaffold → Hopped Molecule
    with the scaffold substructure highlighted in the hopped molecule.

    Parameters
    ----------
    ref_smiles : str
        SMILES of the reference molecule.
    scaffold_smiles : str
        SMILES of the central scaffold.
    hopped_smiles : str
        SMILES of the hopped molecule.
    ref_label : str
        Legend label for the reference molecule.
    scaffold_label : str
        Legend label for the scaffold.
    hopped_label : str
        Legend label for the hopped molecule.
    highlight_color : tuple
        RGB float tuple (0–1) for the highlight color. Default is cornflower blue.
    mol_size : tuple
        (width, height) in pixels for each molecule panel.
    arrow_space : int
        Pixel width reserved for each arrow between panels.
    save_path : str
        File path to save the output PNG. Pass None to skip saving.
    show : bool
        Whether to display the image with matplotlib.

    Returns
    -------
    PIL.Image.Image
        The final composite progression image.
    """

    # ─── Build molecules ────────────────────────────────────────────────
    ref_mol      = Chem.MolFromSmiles(ref_smiles)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    hopped_mol   = Chem.MolFromSmiles(hopped_smiles)

    for mol in [ref_mol, scaffold_mol, hopped_mol]:
        if mol is not None:
            AllChem.Compute2DCoords(mol)

    # ─── MCS-based substructure highlight ───────────────────────────────
    highlight_atoms = []
    highlight_bonds = []

    if scaffold_mol is not None and hopped_mol is not None:

        mcs_result = rdFMCS.FindMCS(
            [scaffold_mol, hopped_mol],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=10,
        )

        if mcs_result.numAtoms > 0:
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            match   = hopped_mol.GetSubstructMatch(mcs_mol)

            if len(match) > 0:
                highlight_atoms = list(match)

                for bond in mcs_mol.GetBonds():
                    a1 = match[bond.GetBeginAtomIdx()]
                    a2 = match[bond.GetEndAtomIdx()]
                    hopped_bond = hopped_mol.GetBondBetweenAtoms(a1, a2)
                    if hopped_bond is not None:
                        highlight_bonds.append(hopped_bond.GetIdx())

        print(f"MCS SMARTS       : {mcs_result.smartsString}")
        print(f"Highlighted atoms: {highlight_atoms}")
        print(f"Highlighted bonds: {highlight_bonds}")

    # ─── Molecule → PIL image ────────────────────────────────────────────
    def mol_to_pil(mol, legend="", h_atoms=None, h_bonds=None):
        h_atoms = h_atoms or []
        h_bonds = h_bonds or []

        drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
        opts = drawer.drawOptions()
        opts.legendFontSize = 22
        opts.bondLineWidth  = 2

        atom_colors = {idx: highlight_color for idx in h_atoms}
        bond_colors = {idx: highlight_color for idx in h_bonds}

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            legend=legend,
            highlightAtoms=h_atoms,
            highlightBonds=h_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        return Image.open(BytesIO(drawer.GetDrawingText()))

    img_ref      = mol_to_pil(ref_mol,      legend=ref_label)
    img_scaffold = mol_to_pil(scaffold_mol,  legend=scaffold_label)
    img_hopped   = mol_to_pil(
        hopped_mol,
        legend=hopped_label,
        h_atoms=highlight_atoms,
        h_bonds=highlight_bonds,
    )

    # ─── Composite canvas ────────────────────────────────────────────────
    w, h = img_ref.size
    canvas_w = w * 3 + arrow_space * 2
    canvas   = Image.new("RGB", (canvas_w, h), (255, 255, 255))

    canvas.paste(img_ref,      (0, 0))
    canvas.paste(img_scaffold, (w + arrow_space, 0))
    canvas.paste(img_hopped,   (2 * (w + arrow_space), 0))

    # ─── Arrows ──────────────────────────────────────────────────────────
    draw  = ImageDraw.Draw(canvas)
    y_mid = h // 2

    def draw_arrow(x_start, x_end):
        draw.line([(x_start, y_mid), (x_end, y_mid)], fill=(0, 0, 0), width=5)
        draw.polygon(
            [(x_end, y_mid), (x_end - 18, y_mid - 10), (x_end - 18, y_mid + 10)],
            fill=(0, 0, 0),
        )

    draw_arrow(w + 10,                    w + arrow_space - 10)
    draw_arrow(2 * w + arrow_space + 10,  2 * w + 2 * arrow_space - 10)

    # ─── Save & show ─────────────────────────────────────────────────────
    if save_path:
        canvas.save(save_path)
        print(f"\nSaved → {save_path}")

    if show:
        plt.figure(figsize=(18, 6))
        plt.imshow(np.array(canvas))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return canvas

