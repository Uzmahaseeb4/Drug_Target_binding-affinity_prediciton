#!/usr/bin/env python
"""
train_gnn.py - train and evaluate the multi-task GNN on your project data
=========================================================================

Reads the same files as the rest of the project
    cdk4_ic50_compounds_unique.csv   her2_ic50_compounds_unique.csv
(SMILES + median IC50 in nM), builds molecular graphs, and trains ONE graph
neural network with two output heads (CDK4, HER2).

Evaluation uses a SCAFFOLD split (whole chemical families held out), so the
test score reflects performance on new chemical series. A Random Forest on
Morgan fingerprints is evaluated on the same split for a fair comparison.

Outputs
-------
gnn_model.pt                 trained model (load with gnn_model.load_bundle)
evaluation_gnn.csv           RMSE / MAE / R2 / Spearman per model and target
predictions_test_gnn.csv     observed vs predicted pIC50 on the test set
eval_pred_vs_actual_gnn.png  scatter plots
gnn_training_curve.png       train / validation loss

Usage
-----
    python train_gnn.py
    python train_gnn.py --epochs 150 --hidden 192
    python train_gnn.py --arch gin
    python train_gnn.py --refit-all      # after evaluating, retrain on ALL data for deployment
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gnn_model import (TASKS, build_label_matrix, predict_smiles, save_bundle,
                       scaffold_split, train_multitask)


def metrics(y, p):
    return {"n": len(y), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p)),
            "Spearman": float(spearmanr(y, p).correlation)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdk4", default="cdk4_ic50_compounds_unique.csv")
    ap.add_argument("--her2", default="her2_ic50_compounds_unique.csv")
    ap.add_argument("--ic50-unit", default="nM", choices=["nM", "uM", "M", "pIC50"])
    ap.add_argument("--arch", default="attentivefp", choices=["attentivefp", "gin"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gnn_model.pt")
    ap.add_argument("--refit-all", action="store_true")
    args = ap.parse_args()
    cfg = dict(arch=args.arch, hidden=args.hidden, layers=args.layers, epochs=args.epochs,
               patience=args.patience, batch=args.batch, lr=args.lr, seed=args.seed)

    # ---- data + scaffold split -------------------------------------------------
    smiles, Y = build_label_matrix(args.cdk4, args.her2, args.ic50_unit)
    tr, te = scaffold_split(smiles, args.test_frac)
    print(f"{len(smiles)} compounds | CDK4 labels {int((~np.isnan(Y[:, 0])).sum())} | "
          f"HER2 labels {int((~np.isnan(Y[:, 1])).sum())} | train {len(tr)} / test {len(te)}")

    # ---- GNN --------------------------------------------------------------------
    print("\nTraining multi-task GNN ...")
    bundle = train_multitask([smiles[i] for i in tr], Y[tr], cfg)
    te_smiles = [smiles[i] for i in te]
    G = predict_smiles(bundle, te_smiles)[["pIC50_CDK4", "pIC50_HER2"]].values

    # ---- Random Forest baseline on the same split ------------------------------------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    FP = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    R = np.full((len(te), 2), np.nan)
    for t in range(2):
        m = tr[~np.isnan(Y[tr, t])]
        rf = RandomForestRegressor(300, max_features=0.3, n_jobs=-1, random_state=args.seed)
        R[:, t] = rf.fit(FP[m], Y[m, t]).predict(FP[te])

    # ---- metrics + files -------------------------------------------------------------
    rows, pred_rows = [], []
    for name, P in [("GNN multi-task", G), ("Random Forest (Morgan)", R)]:
        for t, task in enumerate(TASKS):
            ok = ~np.isnan(Y[te, t])
            rows.append({"model": name, "target": task, **metrics(Y[te][ok, t], P[ok, t])})
            if name.startswith("GNN"):
                for s, o, p in zip(np.array(te_smiles)[ok], Y[te][ok, t], P[ok, t]):
                    pred_rows.append({"smiles": s, "target": task, "observed_pIC50": o,
                                      "predicted_pIC50": p})
    res = pd.DataFrame(rows)
    res.to_csv("evaluation_gnn.csv", index=False)
    pd.DataFrame(pred_rows).to_csv("predictions_test_gnn.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nScaffold-split TEST results:\n", res.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for t, ax in enumerate(axes):
        ok = ~np.isnan(Y[te, t])
        ax.scatter(Y[te][ok, t], G[ok, t], s=12, alpha=.55)
        lim = [min(Y[te][ok, t].min(), G[ok, t].min()), max(Y[te][ok, t].max(), G[ok, t].max())]
        ax.plot(lim, lim, "k--", lw=1)
        r2 = res[(res.model == "GNN multi-task") & (res.target == TASKS[t])]["R2"].iloc[0]
        ax.set(xlabel="Observed pIC50", ylabel="Predicted pIC50",
               title=f"{TASKS[t]} - GNN (test R2 = {r2:.2f})")
    plt.tight_layout()
    plt.savefig("eval_pred_vs_actual_gnn.png", dpi=200)
    plt.close()

    h = np.array(bundle["history"])
    plt.figure(figsize=(5, 3.5))
    plt.plot(h[:, 0], h[:, 1], label="train")
    plt.plot(h[:, 0], h[:, 2], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("masked MSE (standardised)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gnn_training_curve.png", dpi=200)
    plt.close()

    # ---- save ------------------------------------------------------------------------
    if args.refit_all:
        print("\nRetraining on ALL compounds for deployment ...")
        bundle = train_multitask(smiles, Y, cfg)
    save_bundle(bundle, args.model)
    print(f"\nSaved {args.model}, evaluation_gnn.csv, predictions_test_gnn.csv, "
          f"eval_pred_vs_actual_gnn.png, gnn_training_curve.png")


if __name__ == "__main__":
    main()
