#!/usr/bin/env python
"""
gnn_benchmark.py - multi-task GNN vs Random Forest / XGBoost on CDK4 + HER2
===========================================================================

Compares, on IDENTICAL scaffold-grouped cross-validation folds:
    RF                 Random Forest on Morgan bits (one model per target)
    XGBoost            XGBoost on Morgan bits       (one model per target)
    GNN single-task    graph neural network, one model per target
    GNN multi-task     ONE graph neural network with two output heads
                       (CDK4, HER2); the loss ignores missing labels
    Ensemble           average of RF and multi-task GNN

Each SMILES becomes a molecular graph (atoms = nodes, bonds = edges, with RDKit
atom/bond features). Architectures: AttentiveFP (default) or GINE (--arch gin).

Why multi-task + masked loss? Only ~3 compounds were measured on both targets,
so every compound has a label for one head only. The masked loss lets the shared
graph encoder learn from ALL ~2,700 compounds while each head is trained only on
the labels that exist.

Outputs (folder --out)
  benchmark_results.csv   metrics per seed / model / target
  benchmark_summary.csv   mean (+- std over seeds)
  benchmark_r2.png        R2 per model and target
  pred_vs_actual_gnn_mt.png
  gnn_mt_final.pt         multi-task GNN trained on all data (for --predict)

Usage
  pip install torch torch_geometric rdkit scikit-learn pandas matplotlib scipy xgboost
  python gnn_benchmark.py --cdk4 cdk4_ic50_compounds_unique.csv --her2 her2_ic50_compounds_unique.csv
  python gnn_benchmark.py ... --quick                 # small, fast smoke test
  python gnn_benchmark.py ... --seeds 3 --epochs 80   # report mean +- std
  python gnn_benchmark.py --predict "SMILES1" "SMILES2"   # potency + selectivity
"""
import argparse
import copy
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
from torch_geometric.nn.models import AttentiveFP

try:
    from xgboost import XGBRegressor
except ImportError:  # XGBoost is optional
    XGBRegressor = None

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
TASKS = ["CDK4", "HER2"]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _find_col(df, candidates, label):
    for cand in candidates:
        for c in df.columns:
            if cand in c.lower():
                return c
    raise ValueError(f"No {label} column in {list(df.columns)}")


def load_target(path, unit):
    df = pd.read_csv(path)
    s = _find_col(df, ["canonical_smiles", "smiles"], "SMILES")
    v = _find_col(df, ["ic50", "standard_value", "value"], "IC50")
    vals = pd.to_numeric(df[v], errors="coerce")
    if unit == "pIC50":
        p = vals
    else:
        p = {"nM": 9, "uM": 6, "M": 0}[unit] - np.log10(vals.where(vals > 0))
    out = pd.DataFrame({"smiles": df[s], "pic50": p}).dropna()
    out["smiles"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(str(x))) else None
                     for x in out["smiles"]]
    return out.dropna().groupby("smiles", as_index=False)["pic50"].median()


def build_matrix(args):
    tabs = [load_target(args.cdk4, args.ic50_unit), load_target(args.her2, args.ic50_unit)]
    smiles = sorted(set(tabs[0]["smiles"]) | set(tabs[1]["smiles"]))
    pos = {s: i for i, s in enumerate(smiles)}
    Y = np.full((len(smiles), 2), np.nan)
    for j, t in enumerate(tabs):
        Y[[pos[s] for s in t["smiles"]], j] = t["pic50"].values
    return smiles, Y


def scaffold_groups(smiles):
    g = []
    for s in smiles:
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
        except Exception:
            sc = ""
        g.append(sc if sc else s)
    return pd.factorize(pd.Series(g))[0]


# --------------------------------------------------------------------------- #
# Molecular graphs
# --------------------------------------------------------------------------- #
HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3]
BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def onehot(v, choices):
    return [float(v == c) for c in choices] + [float(v not in choices)]


def atom_features(a):
    return (onehot(a.GetAtomicNum(), [6, 7, 8, 16, 9, 17, 35, 53, 15])
            + onehot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
            + onehot(a.GetFormalCharge(), [-1, 0, 1])
            + onehot(a.GetHybridization(), HYB)
            + onehot(a.GetTotalNumHs(), [0, 1, 2, 3])
            + [float(a.GetIsAromatic()), float(a.IsInRing())])


def bond_features(b):
    return onehot(b.GetBondType(), BONDS) + [float(b.GetIsConjugated()), float(b.IsInRing())]


N_ATOM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
N_BOND = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_graph(smi, y_row=None):
    mol = Chem.MolFromSmiles(smi)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        ei += [[i, j], [j, i]]
        ea += [f, f]
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei \
        else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(ea, dtype=torch.float) if ea else torch.zeros((0, N_BOND))
    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y_row is not None:
        d.y = torch.tensor(np.nan_to_num(y_row, nan=0.0), dtype=torch.float).view(1, -1)
        d.mask = torch.tensor(~np.isnan(y_row), dtype=torch.float).view(1, -1)
    return d


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class GINNet(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden, n_out, layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GINEConv(nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
            edge_dim=edge_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(norm(conv(h, edge_index, edge_attr)))
        return self.head(torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], 1))


def make_model(arch, hidden, n_out, layers=3, dropout=0.1):
    if arch == "gin":
        return GINNet(N_ATOM, N_BOND, hidden, n_out, layers, dropout)
    return AttentiveFP(N_ATOM, hidden, n_out, N_BOND, num_layers=layers,
                       num_timesteps=2, dropout=dropout)


def masked_loss(pred, y, mask):
    """Mean squared error over OBSERVED labels only, averaged over tasks."""
    per_task = ((pred - y) ** 2 * mask).sum(0) / mask.sum(0).clamp(min=1)
    return per_task.mean()


def split_train_val(idx, groups, frac, rng):
    g = np.unique(groups[idx])
    val_g = set(rng.choice(g, max(1, int(len(g) * frac)), replace=False))
    is_val = np.array([groups[i] in val_g for i in idx])
    return idx[~is_val], idx[is_val]


def train_gnn(graphs, Y, groups, tr, cols, args, seed):
    """Train on compounds `tr` for task columns `cols`; return (model, mean, std)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = np.array([i for i in tr if not np.all(np.isnan(Y[i, cols]))])
    tr, va = split_train_val(tr, groups, 0.1, rng)
    mean = torch.tensor(np.nanmean(Y[tr][:, cols], 0), dtype=torch.float)
    std = torch.tensor(np.nanstd(Y[tr][:, cols], 0) + 1e-6, dtype=torch.float)
    cols_t = torch.tensor(cols)

    model = make_model(args.arch, args.hidden, len(cols), args.layers, args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    tl = DataLoader([graphs[i] for i in tr], batch_size=args.batch, shuffle=True,
                    drop_last=len(tr) > args.batch)
    vl = DataLoader([graphs[i] for i in va], batch_size=256)

    def batch_loss(b):
        out = model(b.x, b.edge_index, b.edge_attr, b.batch)
        y = (b.y[:, cols_t] - mean) / std
        return masked_loss(out, y, b.mask[:, cols_t])

    best, best_state, bad = np.inf, None, 0
    for _ in range(args.epochs):
        model.train()
        for b in tl:
            opt.zero_grad()
            batch_loss(b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([batch_loss(b).item() for b in vl]))
        sched.step(v)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std


def predict_gnn(model, mean, std, graphs, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for b in DataLoader([graphs[i] for i in idx], batch_size=256):
            out.append(model(b.x, b.edge_index, b.edge_attr, b.batch))
    return (torch.cat(out) * std + mean).numpy()


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def metrics(y, p):
    return {"n": len(y), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p)),
            "Spearman": float(spearmanr(y, p).correlation)}


def run_seed(args, smiles, Y, FP, graphs, groups, seed):
    n = len(smiles)
    names = ["RF", "XGBoost", "GNN single-task", "GNN multi-task"]
    if XGBRegressor is None:
        names.remove("XGBoost")
    preds = {m: np.full((n, 2), np.nan) for m in names}
    folds = list(GroupKFold(args.folds).split(np.zeros(n), groups=groups))

    for f, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        for t in range(2):                                   # tree models, one per target
            m = tr[~np.isnan(Y[tr, t])]
            rf = RandomForestRegressor(args.trees, max_features=0.3, n_jobs=-1,
                                       random_state=seed).fit(FP[m], Y[m, t])
            preds["RF"][te, t] = rf.predict(FP[te])
            if XGBRegressor is not None:
                xg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.4, n_jobs=-1,
                                  tree_method="hist", random_state=seed).fit(FP[m], Y[m, t])
                preds["XGBoost"][te, t] = xg.predict(FP[te])
            mdl, mu, sd = train_gnn(graphs, Y, groups, tr, [t], args, seed + t)
            preds["GNN single-task"][te, t] = predict_gnn(mdl, mu, sd, graphs, te)[:, 0]
        mdl, mu, sd = train_gnn(graphs, Y, groups, tr, [0, 1], args, seed + 10)
        preds["GNN multi-task"][te] = predict_gnn(mdl, mu, sd, graphs, te)
        print(f"  seed {seed} fold {f}/{len(folds)} done in {time.time() - t0:.0f}s")

    preds["Ensemble (RF + GNN-MT)"] = (preds["RF"] + preds["GNN multi-task"]) / 2
    rows = []
    for name, P in preds.items():
        for t, task in enumerate(TASKS):
            ok = ~np.isnan(Y[:, t]) & ~np.isnan(P[:, t])
            rows.append({"seed": seed, "model": name, "target": task,
                         **metrics(Y[ok, t], P[ok, t])})
    return pd.DataFrame(rows), preds


def plot_results(summary, out):
    models = list(dict.fromkeys(summary["model"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.38
    for k, task in enumerate(TASKS):
        s = summary[summary["target"] == task].set_index("model").loc[models]
        ax.bar(np.arange(len(models)) + (k - .5) * w, s["R2_mean"], w,
               yerr=s["R2_std"].fillna(0), label=task, capsize=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("R2 (scaffold-split, out-of-fold)")
    ax.set_title("Model comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "benchmark_r2.png"), dpi=200)
    plt.close()


def plot_scatter(Y, P, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for t, ax in enumerate(axes):
        ok = ~np.isnan(Y[:, t]) & ~np.isnan(P[:, t])
        ax.scatter(Y[ok, t], P[ok, t], s=10, alpha=.5)
        lim = [min(Y[ok, t].min(), P[ok, t].min()), max(Y[ok, t].max(), P[ok, t].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Observed pIC50")
        ax.set_ylabel("Predicted pIC50")
        ax.set_title(f"{TASKS[t]} - multi-task GNN (R2 = {r2_score(Y[ok, t], P[ok, t]):.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pred_vs_actual_gnn_mt.png"), dpi=200)
    plt.close()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdk4", default="cdk4_ic50_compounds_unique.csv")
    ap.add_argument("--her2", default="her2_ic50_compounds_unique.csv")
    ap.add_argument("--ic50-unit", default="nM", choices=["nM", "uM", "M", "pIC50"])
    ap.add_argument("--arch", default="attentivefp", choices=["attentivefp", "gin"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--out", default="gnn_results")
    ap.add_argument("--predict", nargs="+", help="SMILES to score with gnn_mt_final.pt")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.folds, args.hidden, args.trees, args.patience = 15, 3, 64, 100, 5

    # ---- inference mode -----------------------------------------------------
    if args.predict:
        ck = torch.load(os.path.join(args.out, "gnn_mt_final.pt"), weights_only=False)
        model = make_model(ck["arch"], ck["hidden"], 2, ck["layers"], 0.0)
        model.load_state_dict(ck["state"])
        smi = [Chem.MolToSmiles(m) for m in map(Chem.MolFromSmiles, args.predict) if m]
        g = [mol_to_graph(s) for s in smi]
        P = predict_gnn(model, ck["mean"], ck["std"], g, range(len(g)))
        res = pd.DataFrame({"smiles": smi, "pIC50_CDK4": P[:, 0], "pIC50_HER2": P[:, 1],
                            "selectivity (CDK4-HER2)": P[:, 0] - P[:, 1]}).round(2)
        print(res.to_string(index=False))
        return

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    smiles, Y = build_matrix(args)
    print(f"{len(smiles)} compounds | CDK4 labels: {int((~np.isnan(Y[:, 0])).sum())} | "
          f"HER2 labels: {int((~np.isnan(Y[:, 1])).sum())} | "
          f"both: {int((~np.isnan(Y)).all(1).sum())}")
    FP = np.array([_GEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
                  dtype=np.float32)
    graphs = [mol_to_graph(s, Y[i]) for i, s in enumerate(smiles)]
    groups = scaffold_groups(smiles)
    print(f"atom features: {N_ATOM} | bond features: {N_BOND} | scaffold groups: {groups.max() + 1}")

    all_rows, last_preds = [], None
    for sd in range(args.seeds):
        print(f"\n=== seed {sd} ===")
        rows, last_preds = run_seed(args, smiles, Y, FP, graphs, groups, 42 + sd)
        all_rows.append(rows)
    res = pd.concat(all_rows)
    res.to_csv(os.path.join(args.out, "benchmark_results.csv"), index=False)

    summ = (res.groupby(["target", "model"], sort=False)[["RMSE", "MAE", "R2", "Spearman"]]
            .agg(["mean", "std"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(args.out, "benchmark_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nScaffold-split results (out-of-fold):")
    print(summ[["target", "model", "RMSE_mean", "MAE_mean", "R2_mean", "Spearman_mean"]]
          .round(3).to_string(index=False))
    plot_results(summ, args.out)
    plot_scatter(Y, last_preds["GNN multi-task"], args.out)

    # ---- final multi-task GNN on all data -------------------------------------
    print("\nTraining final multi-task GNN on all compounds ...")
    mdl, mu, sd_ = train_gnn(graphs, Y, groups, np.arange(len(smiles)), [0, 1], args, 123)
    torch.save({"state": mdl.state_dict(), "mean": mu, "std": sd_, "arch": args.arch,
                "hidden": args.hidden, "layers": args.layers},
               os.path.join(args.out, "gnn_mt_final.pt"))
    print(f"Saved {args.out}/gnn_mt_final.pt  ->  "
          f"python gnn_benchmark.py --predict \"SMILES\"")


if __name__ == "__main__":
    main()
