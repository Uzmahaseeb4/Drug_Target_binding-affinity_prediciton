#!/usr/bin/env python
"""
gnn_model.py - graph neural network module for the CDK4 / HER2 project
======================================================================

A reusable module (import it from train_gnn.py, predict_affinity.py, app.py ...).

  smiles -> molecular graph (RDKit atom + bond features)
         -> AttentiveFP (or GIN) encoder, shared by both targets
         -> two output heads: pIC50 for CDK4 and pIC50 for HER2

The loss ignores missing labels ("masked"), so compounds measured on only one
target still train the shared encoder.

Quick use from other project files
----------------------------------
    from gnn_model import load_bundle, predict_smiles
    bundle = load_bundle("gnn_model.pt")
    print(predict_smiles(bundle, ["COc1cc2ncnc(Nc3ccccc3)c2cc1"]))

Command line:
    python gnn_model.py --predict "SMILES1" "SMILES2" [--model gnn_model.pt]

Requires: pip install torch torch_geometric rdkit pandas numpy scipy
"""
import argparse
import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
from torch_geometric.nn.models import AttentiveFP

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

TASKS = ["CDK4", "HER2"]
DEFAULT_CFG = dict(arch="attentivefp", hidden=128, layers=3, dropout=0.1,
                   epochs=100, patience=15, batch=64, lr=1e-3, seed=42)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_pic50(path, unit="nM"):
    """Read one project CSV -> DataFrame[smiles, pic50] (canonical SMILES, median per compound).
    Works with cdk4_ic50_compounds_unique.csv (column 'median_ic50_nM')."""
    df = pd.read_csv(path)

    def find(cands, label):
        for cand in cands:
            for c in df.columns:
                if cand in c.lower():
                    return c
        raise ValueError(f"No {label} column in {list(df.columns)}")

    s, v = find(["canonical_smiles", "smiles"], "SMILES"), find(["ic50", "standard_value", "value"], "IC50")
    vals = pd.to_numeric(df[v], errors="coerce")
    p = vals if unit == "pIC50" else {"nM": 9, "uM": 6, "M": 0}[unit] - np.log10(vals.where(vals > 0))
    out = pd.DataFrame({"smiles": df[s], "pic50": p}).dropna()
    out["smiles"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(str(x))) else None
                     for x in out["smiles"]]
    return out.dropna().groupby("smiles", as_index=False)["pic50"].median()


def build_label_matrix(cdk4_path, her2_path, unit="nM"):
    """Union of compounds and an (n, 2) pIC50 matrix with NaN where a label is missing."""
    tabs = [load_pic50(cdk4_path, unit), load_pic50(her2_path, unit)]
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


def scaffold_split(smiles, test_frac=0.2):
    """Deterministic scaffold split: whole scaffold families go to train OR test."""
    groups = scaffold_groups(smiles)
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    train, test = [], []
    n_train_target = (1 - test_frac) * len(smiles)
    for _, idx in sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        (train if len(train) + len(idx) <= n_train_target else test).extend(idx)
    return np.array(sorted(train)), np.array(sorted(test))


# --------------------------------------------------------------------------- #
# Molecular graphs
# --------------------------------------------------------------------------- #
HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3]
BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def _onehot(v, choices):
    return [float(v == c) for c in choices] + [float(v not in choices)]


def atom_features(a):
    return (_onehot(a.GetAtomicNum(), [6, 7, 8, 16, 9, 17, 35, 53, 15])
            + _onehot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
            + _onehot(a.GetFormalCharge(), [-1, 0, 1])
            + _onehot(a.GetHybridization(), HYB)
            + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3])
            + [float(a.GetIsAromatic()), float(a.IsInRing())])


def bond_features(b):
    return _onehot(b.GetBondType(), BONDS) + [float(b.GetIsConjugated()), float(b.IsInRing())]


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
    edge_index = (torch.tensor(ei, dtype=torch.long).t().contiguous() if ei
                  else torch.zeros((2, 0), dtype=torch.long))
    edge_attr = torch.tensor(ea, dtype=torch.float) if ea else torch.zeros((0, N_BOND))
    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y_row is not None:
        d.y = torch.tensor(np.nan_to_num(y_row, nan=0.0), dtype=torch.float).view(1, -1)
        d.mask = torch.tensor(~np.isnan(y_row), dtype=torch.float).view(1, -1)
    return d


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GINNet(nn.Module):
    def __init__(self, hidden, n_out, layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(N_ATOM, hidden)
        self.convs = nn.ModuleList([GINEConv(nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
            edge_dim=N_BOND) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(norm(conv(h, edge_index, edge_attr)))
        return self.head(torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], 1))


def make_model(cfg, n_out=2):
    if cfg["arch"] == "gin":
        return GINNet(cfg["hidden"], n_out, cfg["layers"], cfg["dropout"])
    return AttentiveFP(N_ATOM, cfg["hidden"], n_out, N_BOND, num_layers=cfg["layers"],
                       num_timesteps=2, dropout=cfg["dropout"])


def masked_loss(pred, y, mask):
    """MSE over observed labels only, averaged over the two targets."""
    per_task = ((pred - y) ** 2 * mask).sum(0) / mask.sum(0).clamp(min=1)
    return per_task.mean()


# --------------------------------------------------------------------------- #
# Training / prediction
# --------------------------------------------------------------------------- #
def train_multitask(smiles, Y, cfg=None, verbose=True):
    """Train on the given compounds (Y has NaN for missing labels). Returns a 'bundle'."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    groups = scaffold_groups(smiles)
    ug = np.unique(groups)
    val_g = set(rng.choice(ug, max(1, int(0.1 * len(ug))), replace=False))   # scaffold-separated val
    is_val = np.array([g in val_g for g in groups])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]

    graphs = [mol_to_graph(s, Y[i]) for i, s in enumerate(smiles)]
    mean = torch.tensor(np.nanmean(Y[tr], 0), dtype=torch.float)
    std = torch.tensor(np.nanstd(Y[tr], 0) + 1e-6, dtype=torch.float)

    model = make_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    tl = DataLoader([graphs[i] for i in tr], batch_size=cfg["batch"], shuffle=True,
                    drop_last=len(tr) > cfg["batch"])
    vl = DataLoader([graphs[i] for i in va], batch_size=256)

    def batch_loss(b):
        out = model(b.x, b.edge_index, b.edge_attr, b.batch)
        return masked_loss(out, (b.y - mean) / std, b.mask)

    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tr_loss = []
        for b in tl:
            opt.zero_grad()
            loss = batch_loss(b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            v = float(np.mean([batch_loss(b).item() for b in vl]))
        sched.step(v)
        history.append((ep, float(np.mean(tr_loss)), v))
        if verbose and (ep == 1 or ep % 10 == 0):
            print(f"  epoch {ep:3d} | train loss {np.mean(tr_loss):.3f} | val loss {v:.3f}")
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"  early stop at epoch {ep} (best val loss {best:.3f})")
                break
    model.load_state_dict(best_state)
    return {"state": model.state_dict(), "mean": mean, "std": std, "cfg": cfg, "history": history}


def predict_smiles(bundle, smiles):
    """DataFrame: smiles, pIC50_CDK4, pIC50_HER2, selectivity (CDK4 - HER2)."""
    model = make_model(bundle["cfg"])
    model.load_state_dict(bundle["state"])
    model.eval()
    ok = [i for i, s in enumerate(smiles) if Chem.MolFromSmiles(str(s)) is not None]
    P = np.full((len(smiles), 2), np.nan)
    if ok:
        loader = DataLoader([mol_to_graph(smiles[i]) for i in ok], batch_size=256)
        with torch.no_grad():
            out = torch.cat([model(b.x, b.edge_index, b.edge_attr, b.batch) for b in loader])
        P[ok] = (out * bundle["std"] + bundle["mean"]).numpy()
    if len(ok) < len(smiles):
        print(f"warning: {len(smiles) - len(ok)} invalid SMILES -> NaN")
    return pd.DataFrame({"smiles": list(smiles), "pIC50_CDK4": P[:, 0], "pIC50_HER2": P[:, 1],
                         "selectivity (CDK4-HER2)": P[:, 0] - P[:, 1]})


def save_bundle(bundle, path):
    torch.save(bundle, path)


def load_bundle(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:                      # very old torch without the argument
        return torch.load(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gnn_model.pt")
    ap.add_argument("--predict", nargs="+", required=True)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    print(predict_smiles(load_bundle(a.model), a.predict).round(2).to_string(index=False))
