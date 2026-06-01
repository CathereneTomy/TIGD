
import warnings
warnings.filterwarnings("ignore")

import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from PIL import Image

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA


# ── Colour palette ──────────────────────────────────────────────────────────

PALETTE = (
    "#F4A261", "#E76F51", "#D97706", "#C2410C", "#EAAC8B", "#B56576",
    "#D6D3D1", "#B0B0B0", "#9CA3AF", "#7C7C7C", "#6B7280", "#525252",
    "#4B5563", "#3F3F46", "#374151", "#2F2F2F", "#262626", "#1F2937",
    "#18181B", "#0F0F0F",
)


def colour_for(cluster_id: int) -> str:
    """Return a hex colour for a given cluster ID (cycles through PALETTE)."""
    return PALETTE[cluster_id % len(PALETTE)]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_molecules(
    csv_path: str,
    smiles_col: str = "smiles",
    scaffold_col: str = "scaffold",
) -> tuple[list, list[str], list[str]]:
    """
    Load molecules from a CSV file.

    Parameters
    ----------
    csv_path    : Path to the CSV file.
    smiles_col  : Column name containing SMILES strings.
    scaffold_col: Column name containing scaffold labels.

    Returns
    -------
    mols_all     : List of valid RDKit Mol objects.
    smiles_all   : Corresponding SMILES strings.
    scaffold_all : Corresponding scaffold labels.
    """
    df = pd.read_csv(csv_path)
    mols_all, smiles_all, scaffold_all = [], [], []

    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            continue
        mols_all.append(mol)
        smiles_all.append(str(row[smiles_col]))
        scaffold_all.append(
            str(row[scaffold_col]) if scaffold_col in df.columns else f"mol_{i}"
        )

    print(f"Valid molecules loaded: {len(mols_all)}")
    return mols_all, smiles_all, scaffold_all


# ── Fingerprints ──────────────────────────────────────────────────────────────

def compute_fingerprints(
    mols: list,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[list, np.ndarray]:
    """
    Compute Morgan fingerprints and return both the RDKit objects and a
    numpy bit-matrix.

    Parameters
    ----------
    mols   : List of RDKit Mol objects.
    radius : Morgan radius (default 2).
    n_bits : Fingerprint length (default 2048).

    Returns
    -------
    fps : List of RDKit fingerprint objects.
    X   : (n_mols, n_bits) uint8 numpy array.
    """
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        for mol in mols
    ]
    X = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return fps, X


# ── Butina clustering ─────────────────────────────────────────────────────────

def butina_cluster(
    fps: list,
    similarity_cutoff: float = 0.15,
) -> tuple[list, np.ndarray]:
    """
    Cluster fingerprints with Butina / Taylor-Butina algorithm.

    Parameters
    ----------
    fps               : List of RDKit fingerprint objects.
    similarity_cutoff : Tanimoto similarity threshold (default 0.15).

    Returns
    -------
    clusters       : List of tuples; each tuple contains indices of members.
    cluster_labels : (n_mols,) array mapping each molecule to its cluster ID.
    """
    nfps = len(fps)
    dists = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(
        dists, nfps, 1.0 - similarity_cutoff, isDistData=True
    )
    print(f"Clusters found: {len(clusters)}")

    cluster_labels = np.zeros(nfps, dtype=int)
    for cid, cl in enumerate(clusters):
        for idx in cl:
            cluster_labels[idx] = cid

    return clusters, cluster_labels


# ── Distance matrix ───────────────────────────────────────────────────────────

def compute_distance_matrix(fps: list) -> np.ndarray:
    """
    Build a full pairwise Tanimoto distance matrix.

    Parameters
    ----------
    fps : List of RDKit fingerprint objects.

    Returns
    -------
    dist_matrix : (n, n) float64 numpy array.
    """
    n = len(fps)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dist_matrix[i] = [1.0 - s for s in sims]
    return dist_matrix


# ── Medoids ───────────────────────────────────────────────────────────────────

def get_medoid(indices: list[int], dist_matrix: np.ndarray) -> int:
    """Return the medoid index for a set of molecule indices."""
    if len(indices) == 1:
        return indices[0]
    sub = dist_matrix[np.ix_(indices, indices)]
    return indices[int(np.argmin(sub.mean(axis=1)))]


def get_medoids(clusters: list, dist_matrix: np.ndarray) -> list[int]:
    """
    Compute the medoid (representative) index for every cluster.

    Parameters
    ----------
    clusters     : Cluster list from butina_cluster().
    dist_matrix  : Pairwise distance matrix from compute_distance_matrix().

    Returns
    -------
    List of molecule indices, one per cluster.
    """
    return [get_medoid(list(cl), dist_matrix) for cl in clusters]


# ── Target-cluster selection ──────────────────────────────────────────────────

def find_target_cluster(
    clusters: list,
    central_indices: list[int],
    scaffold_all: list[str],
    smiles_all: list[str],
) -> dict:
    """
    Identify the cluster with the most unique scaffolds.
    Falls back to the largest cluster if every cluster has only one scaffold.

    Parameters
    ----------
    clusters         : Cluster list from butina_cluster().
    central_indices  : Medoid indices from get_medoids().
    scaffold_all     : Scaffold labels aligned with molecule list.
    smiles_all       : SMILES strings aligned with molecule list.

    Returns
    -------
    dict with keys:
        cluster_id, cluster_size, unique_scaffold_count,
        center_smiles, center_scaffold_name
    """
    unique_counts = [
        len({scaffold_all[i] for i in cl}) for cl in clusters
    ]
    max_unique = max(unique_counts)

    if max_unique <= 1:
        print("Fallback: all clusters have only one unique scaffold — selecting largest.")
        target_cid = max(range(len(clusters)), key=lambda c: len(clusters[c]))
    else:
        target_cid = int(np.argmax(unique_counts))

    centre_idx = central_indices[target_cid]

    result = {
        "cluster_id": target_cid,
        "cluster_size": len(clusters[target_cid]),
        "unique_scaffold_count": unique_counts[target_cid],
        "center_smiles": smiles_all[centre_idx],
        "center_scaffold_name": scaffold_all[centre_idx],
    }

    print("\nSelected cluster:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_dimensions(
    X: np.ndarray,
    dist_matrix: np.ndarray,
    method: str = "mds",
) -> tuple[np.ndarray, tuple[str, str]]:
    """
    Reduce fingerprint data to 2-D for visualisation.

    Parameters
    ----------
    X           : Bit-matrix from compute_fingerprints().
    dist_matrix : Pairwise distance matrix from compute_distance_matrix().
    method      : One of 'mds' | 'tsne' | 'pca'.

    Returns
    -------
    X_emb       : (n, 2) embedding coordinates.
    axis_labels : Tuple of (x_label, y_label) strings.
    """
    method = method.lower()
    print(f"Running {method.upper()} dimensionality reduction...")

    if method == "mds":
        model = MDS(
            n_components=2, dissimilarity="precomputed",
            random_state=42, normalized_stress="auto",
            n_init=4, max_iter=500,
        )
        X_emb = model.fit_transform(dist_matrix)
        axis_labels = ("MDS 1", "MDS 2")

    elif method == "tsne":
        perp = min(30, max(2, len(X) // 4))
        model = TSNE(
            n_components=2, perplexity=perp, random_state=42,
            init="pca", learning_rate="auto", n_iter=1000,
        )
        X_emb = model.fit_transform(X)
        axis_labels = ("t-SNE 1", "t-SNE 2")

    elif method == "pca":
        model = PCA(n_components=2, random_state=42)
        X_emb = model.fit_transform(X)
        var = model.explained_variance_ratio_
        axis_labels = (
            f"PC1 ({var[0]*100:.1f}% var)",
            f"PC2 ({var[1]*100:.1f}% var)",
        )

    else:
        raise ValueError(f"Unknown DR method '{method}'. Choose mds | tsne | pca.")

    return X_emb, axis_labels


# ── Scatter plot ──────────────────────────────────────────────────────────────

def plot_scatter(
    X_emb: np.ndarray,
    cluster_labels: np.ndarray,
    clusters: list,
    central_indices: list[int],
    scaffold_all: list[str],
    axis_labels: tuple[str, str],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the chemical-space scatter with convex hulls and cluster centres.

    Parameters
    ----------
    X_emb           : 2-D embedding from reduce_dimensions().
    cluster_labels  : Per-molecule cluster IDs from butina_cluster().
    clusters        : Cluster list from butina_cluster().
    central_indices : Medoid indices from get_medoids().
    scaffold_all    : Scaffold labels.
    axis_labels     : (x_label, y_label) from reduce_dimensions().
    save_path       : If provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#F5F7FA")
    ax.set_facecolor("#FAFBFD")

    point_colours = [colour_for(c) for c in cluster_labels]
    centre_set = set(central_indices)
    nc_idx = [i for i in range(len(X_emb)) if i not in centre_set]

    ax.scatter(
        X_emb[nc_idx, 0], X_emb[nc_idx, 1],
        c=[point_colours[i] for i in nc_idx],
        s=55, alpha=0.72, linewidths=0, zorder=2,
    )

    for cid, idx in enumerate(central_indices):
        col = colour_for(cid)
        ax.scatter(
            X_emb[idx, 0], X_emb[idx, 1],
            s=220, c=col, linewidths=2.0,
            edgecolors="white", zorder=4, marker="D",
        )
        label = scaffold_all[idx]
        if len(label) > 22:
            label = label[:20] + "…"
        ax.text(
            X_emb[idx, 0],
            X_emb[idx, 1] + np.ptp(X_emb[:, 1]) * 0.022,
            label, fontsize=6.5, color="#1A2E4A",
            ha="center", va="bottom", fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="#E8EEF7", ec="#B0BDD0", alpha=0.9),
        )

    # Convex hulls
    try:
        from scipy.spatial import ConvexHull
        for cid, cl in enumerate(clusters):
            pts = X_emb[list(cl)]
            if len(pts) < 3:
                continue
            try:
                hull = ConvexHull(pts)
                verts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(
                    pts[hull.vertices, 0], pts[hull.vertices, 1],
                    color=to_rgba(colour_for(cid), alpha=0.12), zorder=1,
                )
                ax.plot(
                    pts[verts, 0], pts[verts, 1],
                    color=colour_for(cid), lw=0.7, alpha=0.35, zorder=1,
                )
            except Exception:
                pass
    except ImportError:
        pass

    for spine in ax.spines.values():
        spine.set_edgecolor("#D0D7E3")
    ax.grid(True, color="#E2E8F0", lw=0.6)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(
        f"Scaffold Chemical Space · {len(clusters)} clusters",
        fontsize=13, fontweight="bold",
    )

    handles = [
        mpatches.Patch(color=colour_for(i), label=f"C{i} ({len(clusters[i])})")
        for i in range(min(20, len(clusters)))
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved scatter plot → {save_path}")

    return fig


# ── Molecule grid ─────────────────────────────────────────────────────────────

def plot_molecule_grid(
    mols_all: list,
    clusters: list,
    central_indices: list[int],
    scaffold_all: list[str],
    top_n: int | None = None,
    mol_img_size: tuple[int, int] = (300, 300),
    mols_per_row: int = 4,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a grid of cluster-representative molecules.

    Parameters
    ----------
    mols_all        : Full molecule list from load_molecules().
    clusters        : Cluster list from butina_cluster().
    central_indices : Medoid indices from get_medoids().
    scaffold_all    : Scaffold labels.
    top_n           : Show only the top N clusters (None = all).
    mol_img_size    : Pixel size of each molecule tile.
    mols_per_row    : Columns in the grid.
    save_path       : If provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    show_n = len(central_indices) if top_n is None else min(top_n, len(central_indices))
    n_cols = mols_per_row
    n_rows = int(np.ceil(show_n / n_cols))
    mol_px = mol_img_size[0]

    grid_img = Image.new("RGBA", (n_cols * mol_px, n_rows * mol_px), (245, 247, 250, 255))

    for pos, cid in enumerate(range(show_n)):
        idx = central_indices[cid]
        mol_img = Draw.MolToImage(mols_all[idx], size=mol_img_size, kekulize=True).convert("RGBA")

        tile = Image.new("RGBA", mol_img_size, (245, 247, 250, 255))
        bar_col = tuple(int(c * 255) for c in to_rgba(colour_for(cid))[:3]) + (255,)
        tile.paste(mol_img, (0, 0))
        tile.paste(Image.new("RGBA", (mol_px, 6), bar_col), (0, 0))

        row, col_pos = divmod(pos, n_cols)
        grid_img.paste(tile, (col_pos * mol_px, row * mol_px))

    fig, ax = plt.subplots(figsize=(14, 2.8 * n_rows), facecolor="#F5F7FA")
    ax.imshow(np.array(grid_img))
    ax.axis("off")

    for pos, cid in enumerate(range(show_n)):
        idx = central_indices[cid]
        row, col_p = divmod(pos, n_cols)
        cx = (col_p + 0.5) * mol_px

        ax.text(
            cx, row * mol_px + mol_px * 0.06, f"C{cid}",
            color="white", fontsize=8, fontweight="bold",
            ha="center", va="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.25", fc=colour_for(cid), ec="none", alpha=0.92),
        )
        ax.text(
            cx, row * mol_px + mol_px * 0.89,
            textwrap.fill(scaffold_all[idx], width=20),
            fontsize=6.5, ha="center", va="center",
            fontfamily="monospace", color="#1A2E4A",
        )
        ax.text(
            (col_p + 0.88) * mol_px, row * mol_px + mol_px * 0.96,
            f"n={len(clusters[cid])}",
            fontsize=6, color="#6B7A99", ha="center", va="center",
        )

    ax.set_title("Cluster Representative Molecules", fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved molecule grid → {save_path}")

    return fig


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_cluster_summary(
    clusters: list,
    central_indices: list[int],
    smiles_all: list[str],
    scaffold_all: list[str],
    save_path: str,
) -> pd.DataFrame:
    """
    Build and save a CSV summarising every cluster.

    Returns the summary DataFrame.
    """
    rows = []
    for cid, cl in enumerate(clusters):
        idx = central_indices[cid]
        rows.append({
            "cluster_id": cid,
            "size": len(cl),
            "unique_scaffold_count": len({scaffold_all[i] for i in cl}),
            "centre_smiles": smiles_all[idx],
            "centre_scaffold": scaffold_all[idx],
            "member_scaffolds": " | ".join(scaffold_all[i] for i in cl),
        })

    df = pd.DataFrame(rows).sort_values(
        ["unique_scaffold_count", "size"], ascending=False
    )
    df.to_csv(save_path, index=False)
    print(f"Summary CSV → {save_path}")
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_clustering_pipeline(
    protein_name: str,
    csv_path: str,
    smiles_col: str = "smiles",
    scaffold_col: str = "scaffold",
    similarity_cutoff: float = 0.15,
    dr_method: str = "mds",
    mol_img_size: tuple[int, int] = (300, 300),
    mols_per_row: int = 4,
    top_n_clusters: int | None = None,
    show_plots: bool = True,
) -> dict:
    """
    Run the complete scaffold clustering pipeline end-to-end.

    Parameters
    ----------
    csv_path          : Path to the input CSV.
    smiles_col        : SMILES column name.
    scaffold_col      : Scaffold column name.
    similarity_cutoff : Tanimoto similarity cutoff for Butina clustering.
    dr_method         : Dimensionality reduction: 'mds' | 'tsne' | 'pca'.
    mol_img_size      : Size of each molecule image tile in pixels.
    mols_per_row      : Columns in the molecule grid.
    top_n_clusters    : Limit molecule grid to top N clusters (None = all).
    show_plots        : Display figures inline (useful in notebooks).

    Returns
    -------
    dict with keys:
        mols_all, smiles_all, scaffold_all,
        fps, X, clusters, cluster_labels,
        dist_matrix, central_indices,
        target_cluster, df_summary,
        center_smiles  (convenience alias),
        fig_scatter, fig_mols,
        scatter_path, mols_path, summary_path
    """
    
    # 1. Load
    mols_all, smiles_all, scaffold_all = load_molecules(csv_path, smiles_col, scaffold_col)

    # 2. Fingerprints
    fps, X = compute_fingerprints(mols_all)

    # 3. Cluster
    clusters, cluster_labels = butina_cluster(fps, similarity_cutoff)

    # 4. Distance matrix
    dist_matrix = compute_distance_matrix(fps)

    # 5. Medoids
    central_indices = get_medoids(clusters, dist_matrix)

    # 6. Target cluster
    target = find_target_cluster(clusters, central_indices, scaffold_all, smiles_all)

    # 7. Dimensionality reduction
    X_emb, axis_labels = reduce_dimensions(X, dist_matrix, dr_method)

    # 8. Scatter plot
    scatter_path = f"{protein_name}/images/{protein_name}_scatter_{dr_method}.png"
    fig_scatter = plot_scatter(
        X_emb, cluster_labels, clusters, central_indices,
        scaffold_all, axis_labels, save_path=scatter_path,
    )

    # 9. Molecule grid
    mols_path = f"{protein_name}/images/{protein_name}_cluster_centres.png"
    fig_mols = plot_molecule_grid(
        mols_all, clusters, central_indices, scaffold_all,
        top_n=top_n_clusters, mol_img_size=mol_img_size,
        mols_per_row=mols_per_row, save_path=mols_path,
    )

    # 10. Summary CSV
    summary_path = f"{protein_name}/csvs/{protein_name}_cluster_summary.csv"
    df_summary = save_cluster_summary(
        clusters, central_indices, smiles_all, scaffold_all, summary_path
    )

    if show_plots:
        plt.show()

    return {
        "mols_all": mols_all,
        "smiles_all": smiles_all,
        "scaffold_all": scaffold_all,
        "fps": fps,
        "X": X,
        "clusters": clusters,
        "cluster_labels": cluster_labels,
        "dist_matrix": dist_matrix,
        "central_indices": central_indices,
        "target_cluster": target,
        "center_smiles": target["center_smiles"],
        "center_scaffold_name": target["center_scaffold_name"],   # convenience alias
        "df_summary": df_summary,
        "fig_scatter": fig_scatter,
        "fig_mols": fig_mols,
        "scatter_path": scatter_path,
        "mols_path": mols_path,
        "summary_path": summary_path,
    }