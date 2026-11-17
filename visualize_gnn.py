#!/usr/bin/env python
"""
visualize_gnn.py - see what the GNN is doing
============================================

Uses the model saved by train_gnn.py (gnn_model.pt) and produces three kinds of
figures in the folder gnn_viz/ :

1. graph_<n>.png            HOW A MOLECULE BECOMES A GRAPH
                            left: normal structure; right: the graph the GNN sees
                            (nodes = atoms, edges = bonds) with tensor shapes.

2. atom_importance_<n>.png  WHICH ATOMS DRIVE EACH PREDICTION
                            atoms coloured for the CDK4 head and the HER2 head
                            (red = raises predicted pIC50, blue = lowers it).
                            --method occlusion (default): blank out one atom's
                                features and measure how much the prediction changes.
                            --method gradient: gradient x input saliency.

3. embedding_tsne.png       WHAT THE MODEL LEARNED OVERALL
                            the molecule representation just before the output
                            layer, flattened to 2-D with t-SNE, coloured by target
                            and by observed pIC50. Nearby points = molecules the
                            network treats as similar.

Usage
-----
    python visualize_gnn.py                               # top compounds from your CSVs
    python visualize_gnn.py --smiles "COc1cc2ncnc(Nc3ccccc3)c2cc1"
    python visualize_gnn.py --method gradient --top 5
    python visualize_gnn.py --no-embedding                # skip the slower t-SNE plot
"""
import argparse
import io
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.loader import DataLoader

from gnn_model import (TASKS, N_ATOM, N_BOND, build_label_matrix, load_bundle, make_model,
                       mol_to_graph)

warnings.filterwarnings("ignore")
ELEMENT_COLORS = {"C": "#555555", "N": "#2f5fd0", "O": "#d03030", "S": "#d0a000",
                  "F": "#2aa84a", "Cl": "#2aa84a", "Br": "#8a4b2a", "P": "#d07000"}


def load_model(path):
    bundle = load_bundle(path)
    model = make_model(bundle["cfg"])
    model.load_state_dict(bundle["state"])
    model.eval()
    return model, bundle


# --------------------------------------------------------------------------- #
# 1. Molecule -> graph picture
# --------------------------------------------------------------------------- #
def draw_graph_view(smi, path):
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)
    pos = mol.GetConformer().GetPositions()[:, :2]
    g = mol_to_graph(smi)

    d = rdMolDraw2D.MolDraw2DCairo(520, 420)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    img = Image.open(io.BytesIO(d.GetDrawingText()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Molecule (SMILES -> structure)", fontsize=11)

    ax = axes[1]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ax.plot(*zip(pos[i], pos[j]), color="#888888", lw=1.5 + 1.5 * (b.GetBondTypeAsDouble() > 1.4),
                zorder=1)
    for a in mol.GetAtoms():
        i, sym = a.GetIdx(), a.GetSymbol()
        ax.scatter(*pos[i], s=420, color=ELEMENT_COLORS.get(sym, "#777777"), zorder=2,
                   edgecolor="white", linewidth=1.5)
        ax.text(*pos[i], f"{sym}\n{i}", ha="center", va="center", color="white",
                fontsize=7, fontweight="bold", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Graph the GNN sees (node = atom, edge = bond)", fontsize=11)
    fig.text(0.5, 0.02,
             f"x: [{g.x.shape[0]} atoms, {N_ATOM} features]   "
             f"edge_index: [2, {g.edge_index.shape[1]}]   "
             f"edge_attr: [{g.edge_attr.shape[0]}, {N_BOND}]",
             ha="center", fontsize=10, family="monospace")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Atom importance
# --------------------------------------------------------------------------- #
def atom_scores(model, bundle, smi, head, method):
    """Per-atom contribution to the (de-standardised) pIC50 of one output head."""
    g = mol_to_graph(smi)
    n = g.x.shape[0]
    batch = torch.zeros(n, dtype=torch.long)
    std, mean = bundle["std"][head], bundle["mean"][head]

    def predict(x):
        return model(x, g.edge_index, g.edge_attr, batch)[0, head] * std + mean

    if method == "gradient":
        x = g.x.clone().requires_grad_(True)
        predict(x).backward()
        return ((x.grad * g.x).sum(1) * std).detach().numpy()

    with torch.no_grad():                                   # occlusion
        full = predict(g.x)
        scores = []
        for i in range(n):
            x = g.x.clone()
            x[i] = 0
            scores.append(float(full - predict(x)))
    return np.array(scores)


def predicted_pic50(model, bundle, smi):
    g = mol_to_graph(smi)
    with torch.no_grad():
        out = model(g.x, g.edge_index, g.edge_attr, torch.zeros(g.x.shape[0], dtype=torch.long))
    return (out[0] * bundle["std"] + bundle["mean"]).numpy()


def heat_image(mol, w, size=(480, 380)):
    vmax = max(np.abs(w).max(), 1e-9)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = matplotlib.colormaps["bwr"]
    hl = [i for i in range(len(w)) if abs(w[i]) > 0.08 * vmax]
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=hl,
        highlightAtomColors={i: tuple(cmap(norm(w[i]))[:3]) for i in hl},
        highlightAtomRadii={i: 0.45 for i in hl})
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def draw_importance(model, bundle, smi, method, path):
    mol = Chem.MolFromSmiles(smi)
    pred = predicted_pic50(model, bundle, smi)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for h, ax in enumerate(axes):
        ax.imshow(heat_image(mol, atom_scores(model, bundle, smi, h, method)))
        ax.axis("off")
        ax.set_title(f"{TASKS[h]}: predicted pIC50 = {pred[h]:.2f}", fontsize=11)
    fig.suptitle(smi if len(smi) < 70 else smi[:67] + "...", fontsize=9)
    fig.text(0.5, 0.01, f"red = raises predicted pIC50  |  blue = lowers it   "
                        f"(method: {method})", ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Embedding map
# --------------------------------------------------------------------------- #
def embeddings(model, smiles):
    """Molecule representation entering the last linear layer (one row per SMILES)."""
    last = model.lin2 if hasattr(model, "lin2") else model.head[-1]
    store = []
    hook = last.register_forward_pre_hook(lambda m, inp: store.append(inp[0].detach()))
    loader = DataLoader([mol_to_graph(s) for s in smiles], batch_size=256)
    with torch.no_grad():
        for b in loader:
            model(b.x, b.edge_index, b.edge_attr, b.batch)
    hook.remove()
    return torch.cat(store).numpy()


def draw_embedding(model, smiles, Y, path, seed=42):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    E = StandardScaler().fit_transform(embeddings(model, smiles))
    perp = min(30, max(5, len(smiles) // 4))
    Z = TSNE(2, perplexity=perp, init="pca", random_state=seed).fit_transform(E)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    has = ~np.isnan(Y)
    cat = np.where(has.all(1), 2, np.where(has[:, 0], 0, 1))
    for c, (name, col) in enumerate([("CDK4 only", "#0e7c86"), ("HER2 only", "#f26b4f"),
                                      ("both", "#222222")]):
        axes[0].scatter(*Z[cat == c].T, s=8, alpha=.7, color=col, label=name)
    axes[0].legend(markerscale=2)
    axes[0].set_title("Coloured by which target was measured")
    for t in range(2):
        ax = axes[t + 1]
        ax.scatter(*Z[~has[:, t]].T, s=6, color="#dddddd")
        sc = ax.scatter(*Z[has[:, t]].T, s=9, c=Y[has[:, t], t], cmap="viridis")
        plt.colorbar(sc, ax=ax, label="observed pIC50")
        ax.set_title(f"Coloured by {TASKS[t]} pIC50")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gnn_model.pt")
    ap.add_argument("--cdk4", default="cdk4_ic50_compounds_unique.csv")
    ap.add_argument("--her2", default="her2_ic50_compounds_unique.csv")
    ap.add_argument("--ic50-unit", default="nM", choices=["nM", "uM", "M", "pIC50"])
    ap.add_argument("--smiles", nargs="+", help="molecules to explain")
    ap.add_argument("--top", type=int, default=3, help="most potent compounds per target")
    ap.add_argument("--method", default="occlusion", choices=["occlusion", "gradient"])
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default="gnn_viz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, bundle = load_model(args.model)
    smiles_all = Y = None
    if not (args.smiles and args.no_embedding):
        try:
            smiles_all, Y = build_label_matrix(args.cdk4, args.her2, args.ic50_unit)
        except FileNotFoundError as e:
            print(f"(project CSVs not found: {e}); only --smiles figures will be made)")

    if args.smiles:
        targets = [Chem.MolToSmiles(m) for m in map(Chem.MolFromSmiles, args.smiles) if m]
    elif smiles_all is not None:
        targets = []
        for t in range(2):
            order = np.argsort(-np.nan_to_num(Y[:, t], nan=-1))[:args.top]
            targets += [smiles_all[i] for i in order]
    else:
        raise SystemExit("Give --smiles or make the project CSVs available.")

    for n, smi in enumerate(targets, 1):
        draw_graph_view(smi, os.path.join(args.out, f"graph_{n}.png"))
        draw_importance(model, bundle, smi, args.method,
                        os.path.join(args.out, f"atom_importance_{n}.png"))
        print(f"[{n}/{len(targets)}] {smi[:60]}")

    if not args.no_embedding and smiles_all is not None:
        print("Computing t-SNE of learned embeddings ...")
        draw_embedding(model, smiles_all, Y, os.path.join(args.out, "embedding_tsne.png"))
    print(f"\nFigures saved in {args.out}/")


if __name__ == "__main__":
    main()
