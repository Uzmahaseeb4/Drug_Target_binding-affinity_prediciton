#!/usr/bin/env python
"""
SHAP interpretation of the CDK4 / HER2 models (Morgan bits -> substructures)
============================================================================

Works on the models saved by dual_target_selectivity.py (dual_models.joblib).

Produces, in the output folder:
  1. GLOBAL, per target (CDK4, HER2)
       shap_global_<T>.csv          top bits: mean |SHAP|, direction, example substructure
       shap_global_<T>_bar.png      bar chart of the most influential substructures
       shap_global_<T>_bits.png     the substructures drawn on example molecules
  2. SELECTIVITY (CDK4 vs HER2)
       shap_selectivity.csv         bits that push a compound toward CDK4 or toward HER2
       shap_selectivity_bar.png / shap_selectivity_bits.png
  3. LOCAL, per molecule
       explain_<n>.png              molecule with atoms coloured by SHAP contribution,
                                    CDK4 panel next to HER2 panel
                                    (red = raises predicted pIC50, blue = lowers it)

Usage
-----
  python shap_interpretation.py --model dual_models.joblib
  python shap_interpretation.py --model dual_models.joblib --smiles "COc1cc2ncnc(Nc3ccccc3)c2cc1"
  python shap_interpretation.py --model dual_models.joblib --ranking dual_results/dual_ranking.csv --top 8

Needs:  pip install shap rdkit scikit-learn pandas matplotlib pillow joblib
"""
import argparse
import io
import os
import warnings
from collections import Counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import colors
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# Must match dual_target_selectivity.py
FP_RADIUS, FP_BITS = 2, 2048
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


# --------------------------------------------------------------------------- #
# Fingerprint + bit -> atoms mapping
# --------------------------------------------------------------------------- #
def fp_and_info(mol):
    """Fingerprint (0/1 array) and bitInfo {bit: ((atom, radius), ...)}."""
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = _GEN.GetFingerprintAsNumPy(mol, additionalOutput=ao)
    return fp, dict(ao.GetBitInfoMap())


def env_atoms_bonds(mol, atom, radius):
    """Atoms and bonds in the circular environment of `atom` with given radius."""
    if radius == 0:
        return {atom}, []
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom))
    atoms = {atom}
    for b in bonds:
        bd = mol.GetBondWithIdx(b)
        atoms.update([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()])
    return atoms, bonds


def fragment_smiles(mol, atom, radius):
    atoms, bonds = env_atoms_bonds(mol, atom, radius)
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms),
                                        bondsToUse=bonds, rootedAtAtom=atom)
    except Exception:
        return mol.GetAtomWithIdx(atom).GetSymbol()


def describe_bit(bit, mols, infos):
    """Most common substructure behind a bit (+ one example molecule to draw it on).
    Note: with 2048 bits, hash collisions are possible; the most frequent
    environment is reported."""
    counts, example = Counter(), {}
    for mol, info in zip(mols, infos):
        if bit in info:
            atom, rad = info[bit][0]
            frag = fragment_smiles(mol, atom, rad)
            counts[frag] += 1
            example.setdefault(frag, (mol, info))
    if not counts:
        return None, None, None
    frag = counts.most_common(1)[0][0]
    return frag, example[frag][0], example[frag][1]


def atom_contributions(mol, shap_row, info):
    """Distribute each present bit's SHAP value over the atoms of its environment."""
    w = np.zeros(mol.GetNumAtoms())
    for bit, occ in info.items():
        per_occ = shap_row[bit] / len(occ)
        for atom, rad in occ:
            atoms, _ = env_atoms_bonds(mol, atom, rad)
            for a in atoms:
                w[a] += per_occ / len(atoms)
    return w


# --------------------------------------------------------------------------- #
# SHAP wrapper (average of the CV fold models)
# --------------------------------------------------------------------------- #
class EnsembleExplainer:
    def __init__(self, models):
        self.models = models
        self.explainers = [shap.TreeExplainer(m) for m in models]

    def shap_values(self, X):
        return np.mean([e.shap_values(X, check_additivity=False)
                        for e in self.explainers], axis=0)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    @property
    def base_value(self):
        return float(np.mean([np.ravel(e.expected_value)[0] for e in self.explainers]))


def prep(smiles_list):
    mols, X, infos = [], [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp, info = fp_and_info(mol)
        mols.append(mol)
        X.append(fp)
        infos.append(info)
    return mols, np.array(X), infos


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def bar_plot(df, value_col, title, xlabel, path, pos_label="raises", neg_label="lowers"):
    d = df.iloc[::-1]
    cols = ["#c0392b" if v > 0 else "#2e6fb5" for v in d[value_col]]
    plt.figure(figsize=(7.5, 0.38 * len(d) + 1.5))
    plt.barh([f"bit {b}  {f}" for b, f in zip(d["bit"], d["substructure"])],
             d[value_col], color=cols)
    plt.axvline(0, color="k", lw=.8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def bits_grid(rows, path, legends):
    """Draw the substructures (highlighted on example molecules)."""
    tpls = [(m, int(b), info) for (b, m, info) in rows]
    try:
        img = Draw.DrawMorganBits(tpls, molsPerRow=4, subImgSize=(220, 220),
                                  legends=legends, useSVG=False)
        if isinstance(img, bytes):          # newer RDKit: raw PNG bytes
            with open(path, "wb") as fh:
                fh.write(img)
        elif isinstance(img, str):          # SVG text
            with open(path.replace(".png", ".svg"), "w") as fh:
                fh.write(img)
        else:                               # PIL image (older RDKit)
            img.save(path)
    except Exception as e:  # drawing is cosmetic - never kill the run
        print(f"  (could not draw bit grid: {e})")


def draw_atom_heatmap(mol, w, size=(480, 380)):
    """Molecule with atoms highlighted by contribution: red (+) / blue (-)."""
    vmax = max(np.abs(w).max(), 1e-9)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = matplotlib.colormaps["bwr"]
    hl = [i for i in range(len(w)) if abs(w[i]) > 0.08 * vmax]
    cols = {i: tuple(cmap(norm(w[i]))[:3]) for i in hl}
    rad = {i: 0.45 for i in hl}
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hl,
                                       highlightAtomColors=cols, highlightAtomRadii=rad)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


# --------------------------------------------------------------------------- #
# 1. Global importance per target
# --------------------------------------------------------------------------- #
def global_analysis(name, target, expl, out, max_samples, top_n, rng):
    smiles = target["smiles"]
    idx = rng.choice(len(smiles), min(max_samples, len(smiles)), replace=False)
    mols, X, infos = prep([smiles[i] for i in idx])
    print(f"[{name}] computing SHAP for {len(mols)} compounds ...")
    sv = expl.shap_values(X)

    present = X.astype(bool)
    n_pres = present.sum(0)
    mean_abs = np.abs(sv).mean(0)
    mean_when = (sv * present).sum(0) / np.maximum(n_pres, 1)

    rows, recs = [], []
    for bit in np.argsort(-mean_abs)[:top_n]:
        if n_pres[bit] == 0:
            continue
        frag, mol, info = describe_bit(bit, mols, infos)
        if frag is None:
            continue
        rows.append((bit, mol, info))
        recs.append({"bit": int(bit), "substructure": frag,
                     "mean_abs_shap": mean_abs[bit], "n_present": int(n_pres[bit]),
                     "mean_shap_when_present": mean_when[bit],
                     "effect": "raises pIC50" if mean_when[bit] > 0 else "lowers pIC50"})
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, f"shap_global_{name}.csv"), index=False)

    bar_plot(df.head(15), "mean_shap_when_present",
             f"{name}: substructures that move predicted pIC50",
             "mean SHAP when present (pIC50 units; red = raises, blue = lowers)",
             os.path.join(out, f"shap_global_{name}_bar.png"))
    bits_grid(rows[:12], os.path.join(out, f"shap_global_{name}_bits.png"),
              [f"bit {r['bit']} | {r['mean_shap_when_present']:+.2f}" for r in recs[:12]])
    return df


# --------------------------------------------------------------------------- #
# 2. Selectivity drivers (CDK4 SHAP - HER2 SHAP)
# --------------------------------------------------------------------------- #
def selectivity_analysis(cdk4, her2, e4, e2, out, max_samples, top_n, min_count, rng):
    pool = sorted(set(cdk4["smiles"]) | set(her2["smiles"]))
    idx = rng.choice(len(pool), min(max_samples, len(pool)), replace=False)
    mols, X, infos = prep([pool[i] for i in idx])
    print(f"[selectivity] computing SHAP for {len(mols)} compounds on both targets ...")
    diff = e4.shap_values(X) - e2.shap_values(X)

    present = X.astype(bool)
    n_pres = present.sum(0)
    mean_diff = (diff * present).sum(0) / np.maximum(n_pres, 1)
    ok = n_pres >= min_count
    order = np.argsort(-np.where(ok, mean_diff, -np.inf))
    cdk_bits = [b for b in order[:top_n] if ok[b] and mean_diff[b] > 0]
    order = np.argsort(np.where(ok, mean_diff, np.inf))
    her_bits = [b for b in order[:top_n] if ok[b] and mean_diff[b] < 0]

    recs, rows = [], []
    for label, bits in [("favours CDK4", cdk_bits), ("favours HER2", her_bits)]:
        for bit in bits:
            frag, mol, info = describe_bit(bit, mols, infos)
            if frag is None:
                continue
            recs.append({"bit": int(bit), "substructure": frag, "direction": label,
                         "mean_selectivity_shap": mean_diff[bit],
                         "n_present": int(n_pres[bit])})
            rows.append((bit, mol, info))
    if not recs:
        print("[selectivity] no bits met the minimum-count threshold - lower --min-count")
        return None
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, "shap_selectivity.csv"), index=False)
    bar_plot(df, "mean_selectivity_shap",
             "Substructures driving selectivity",
             "mean (SHAP_CDK4 - SHAP_HER2): red -> favours CDK4, blue -> favours HER2",
             os.path.join(out, "shap_selectivity_bar.png"))
    bits_grid(rows, os.path.join(out, "shap_selectivity_bits.png"),
              [f"{r['direction']} | {r['mean_selectivity_shap']:+.2f}" for r in recs])
    return df


# --------------------------------------------------------------------------- #
# 3. Local explanation: molecule highlighted for CDK4 and HER2
# --------------------------------------------------------------------------- #
def explain_molecules(smiles_list, e4, e2, out):
    for n, smi in enumerate(smiles_list, 1):
        mols, X, infos = prep([smi])
        if not mols:
            print(f"  skipped invalid SMILES: {smi}")
            continue
        mol, info = mols[0], infos[0]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, name, ex in zip(axes, ["CDK4", "HER2"], [e4, e2]):
            sv = ex.shap_values(X)[0]
            w = atom_contributions(mol, sv, info)
            ax.imshow(draw_atom_heatmap(mol, w))
            ax.axis("off")
            ax.set_title(f"{name}: predicted pIC50 = {ex.predict(X)[0]:.2f}\n"
                         f"(model baseline {ex.base_value:.2f})", fontsize=10)
        fig.suptitle(smi if len(smi) < 70 else smi[:67] + "...", fontsize=9)
        fig.text(0.5, 0.01, "red = raises predicted pIC50   |   blue = lowers it",
                 ha="center", fontsize=9)
        plt.tight_layout(rect=(0, 0.04, 1, 0.95))
        path = os.path.join(out, f"explain_{n}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  saved {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="dual_models.joblib")
    ap.add_argument("--out", default="shap_results")
    ap.add_argument("--smiles", nargs="+", help="molecules to explain")
    ap.add_argument("--ranking", help="dual_ranking.csv: explain its top compounds")
    ap.add_argument("--top", type=int, default=6, help="how many molecules to explain")
    ap.add_argument("--max-samples", type=int, default=200,
                    help="compounds used for global SHAP (more = slower)")
    ap.add_argument("--top-bits", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=5,
                    help="min compounds containing a bit for selectivity analysis")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bundle = joblib.load(args.model)
    cdk4, her2 = bundle["CDK4"], bundle["HER2"]
    e4, e2 = EnsembleExplainer(cdk4["models"]), EnsembleExplainer(her2["models"])

    # 1. global
    for name, tgt, ex in [("CDK4", cdk4, e4), ("HER2", her2, e2)]:
        df = global_analysis(name, tgt, ex, args.out, args.max_samples, args.top_bits, rng)
        print(f"\nTop substructures for {name}:")
        print(df[["bit", "substructure", "mean_abs_shap", "mean_shap_when_present"]]
              .head(8).round(3).to_string(index=False))

    # 2. selectivity
    sel = selectivity_analysis(cdk4, her2, e4, e2, args.out, args.max_samples,
                               8, args.min_count, rng)
    if sel is not None:
        print("\nSelectivity drivers:")
        print(sel.round(3).to_string(index=False))

    # 3. local
    if args.smiles:
        targets = args.smiles
    elif args.ranking:
        targets = pd.read_csv(args.ranking)["smiles"].head(args.top).tolist()
    else:  # most potent known compounds for each target
        t4 = pd.Series(cdk4["y"], index=cdk4["smiles"]).nlargest(args.top // 2).index.tolist()
        t2 = pd.Series(her2["y"], index=her2["smiles"]).nlargest(args.top // 2).index.tolist()
        targets = t4 + t2
    print("\nExplaining individual molecules ...")
    explain_molecules(targets, e4, e2, args.out)
    print(f"\nDone. Results in: {args.out}/")


if __name__ == "__main__":
    main()
