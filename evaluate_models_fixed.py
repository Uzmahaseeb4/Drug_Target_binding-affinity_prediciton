"""
Robust evaluation for drug–protein pIC50 prediction models.

Inputs:
  predictions_test.csv (preferred; must contain molecule_chembl_id, target,
                        y_true and one or more model prediction columns)
  OR, if missing / --retrain is supplied:
      X_pairs.npy, y_pIC50.npy, pair_dataset.csv

Outputs:
  predictions_test.csv (when generated)
  evaluation_overall.csv
  evaluation_per_target.csv
  eval_pred_vs_actual.png
  eval_residuals.png
  eval_metric_bars.png

Usage:
  python evaluate_models.py
  python evaluate_models.py --retrain

Notes:
- --retrain regenerates predictions so an old predictions_test.csv is not reused.
- The fallback uses a molecule-grouped split. If pair_dataset.csv contains a
  usable SMILES column and RDKit is installed, it attempts a Bemis–Murcko
  scaffold-grouped split instead.
- Model-selection decisions should be made on validation data, not on the final
  test set. This script reports test-set metrics; it does not tune models.
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_NAMES = ["RandomForest", "XGBoost", "SVR"]
REQUIRED_PREDICTION_COLUMNS = ["molecule_chembl_id", "target", "y_true"]


def safe_corr(func, a, b):
    """Return NaN when correlation is undefined (e.g., constant input)."""
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    try:
        result = func(a, b)
        return float(result.statistic if hasattr(result, "statistic") else result[0])
    except (ValueError, FloatingPointError):
        return float("nan")


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def score(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) == 0:
        return {
            "N": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
            "Pearson_r": np.nan, "Spearman_rho": np.nan,
            "Within_0.5_log": np.nan, "Within_1.0_log": np.nan,
        }
    return {
        "N": int(len(a)),
        "R2": float(r2_score(a, b)) if len(a) >= 2 and np.unique(a).size > 1 else np.nan,
        "RMSE": rmse(a, b),
        "MAE": float(mean_absolute_error(a, b)),
        "Pearson_r": safe_corr(pearsonr, a, b),
        "Spearman_rho": safe_corr(spearmanr, a, b),
        "Within_0.5_log": float((np.abs(a - b) <= 0.5).mean()),
        "Within_1.0_log": float((np.abs(a - b) <= 1.0).mean()),
    }


def bootstrap_ci(a, b, groups=None, n_boot=1000, seed=42):
    """
    Percentile 95% CIs for R2 and RMSE.
    If groups are supplied, resample molecule groups rather than individual rows.
    R2 replicates with constant y_true are skipped.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if groups is not None:
        groups = np.asarray(groups)[valid]

    if len(a) < 2:
        return (np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))

    rng = np.random.default_rng(seed)
    r2s, rmses = [], []

    if groups is None:
        units = np.arange(len(a))
        row_indices = {i: np.array([i]) for i in units}
    else:
        unique_groups = pd.unique(groups)
        row_indices = {
            group: np.flatnonzero(groups == group) for group in unique_groups
        }
        units = np.asarray(unique_groups, dtype=object)

    for _ in range(n_boot):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([row_indices[u] for u in sampled_units])
        aa, bb = a[idx], b[idx]
        if len(aa) == 0:
            continue
        rmses.append(rmse(aa, bb))
        if len(aa) >= 2 and np.unique(aa).size > 1:
            value = r2_score(aa, bb)
            if np.isfinite(value):
                r2s.append(value)

    r2_ci = np.percentile(r2s, [2.5, 97.5]) if r2s else np.array([np.nan, np.nan])
    rmse_ci = np.percentile(rmses, [2.5, 97.5]) if rmses else np.array([np.nan, np.nan])
    return r2_ci, rmse_ci


def make_predictions(output_path="predictions_test.csv", split_mode="auto"):
    """Train baseline models and create a fresh held-out prediction file."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    for filename in ["X_pairs.npy", "y_pIC50.npy", "pair_dataset.csv"]:
        if not os.path.exists(filename):
            sys.exit(f"{filename} not found. Run the feature-concatenation step first.")

    X = np.load("X_pairs.npy")
    y = np.load("y_pIC50.npy").reshape(-1)
    meta = pd.read_csv("pair_dataset.csv")

    if not {"molecule_chembl_id", "target"}.issubset(meta.columns):
        sys.exit("pair_dataset.csv must contain molecule_chembl_id and target.")
    if len(X) != len(y) or len(y) != len(meta):
        sys.exit(
            f"Row mismatch: X={len(X)}, y={len(y)}, metadata={len(meta)}. "
            "Regenerate aligned feature and metadata files."
        )
    if X.ndim != 2:
        sys.exit(f"X_pairs.npy must be 2-D; received shape {X.shape}.")
    if not np.isfinite(X).all():
        sys.exit("X_pairs.npy contains NaN or infinite values.")
    if not np.isfinite(y).all():
        sys.exit("y_pIC50.npy contains NaN or infinite values.")

    # Prefer scaffold grouping when a plausible SMILES column and RDKit are available.
    smiles_col = next(
        (c for c in ["canonical_smiles", "smiles", "SMILES", "canonical_SMILES"]
         if c in meta.columns),
        None,
    )
    groups = meta["molecule_chembl_id"].astype(str).to_numpy()
    split_description = "molecule-grouped split"

    if split_mode != "molecule" and smiles_col:
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            scaffolds = []
            for smi in meta[smiles_col].fillna("").astype(str):
                mol = Chem.MolFromSmiles(smi)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""
                scaffolds.append(scaffold if scaffold else f"INVALID_OR_ACYCLIC_{smi}")
            scaffold_groups = np.asarray(scaffolds, dtype=object)
            if len(pd.unique(scaffold_groups)) >= 2:
                groups = scaffold_groups
                split_description = "Bemis–Murcko scaffold-grouped split"
            else:
                warnings.warn("Not enough distinct scaffolds; using molecule-grouped split.")
        except ImportError:
            if split_mode == "scaffold":
                sys.exit("RDKit is required for --split scaffold. Install it with conda install -c conda-forge rdkit.")
            warnings.warn("RDKit not installed; falling back to molecule-grouped split.")
    elif split_mode == "scaffold":
        sys.exit(
            "No SMILES column found in pair_dataset.csv. Add a canonical_smiles/smiles "
            "column or use --split molecule."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(splitter.split(X, y, groups=groups))
    print(f"Using {split_description}: train={len(tr)}, test={len(te)}")

    # Feature-layout check for the SVR scaler: original design assumes 2048
    # fingerprint bits followed by protein features.
    if X.shape[1] <= 2048:
        warnings.warn(
            f"X has {X.shape[1]} features. SVR will scale all features because "
            "there is no feature block after column 2048."
        )
        svr_transformer = StandardScaler()
    else:
        svr_transformer = ColumnTransformer(
            [("protein", StandardScaler(), slice(2048, None))],
            remainder="passthrough",
        )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, tree_method="hist",
            random_state=42, n_jobs=-1,
        ),
        "SVR": Pipeline([
            ("scale", svr_transformer),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
        ]),
    }

    out = meta.iloc[te].reset_index(drop=True)[["molecule_chembl_id", "target"]].copy()
    out["y_true"] = y[te]
    out["split_group"] = groups[te]
    for name, model in models.items():
        print(f"Fitting {name} ...")
        model.fit(X[tr], y[tr])
        out[name] = model.predict(X[te])

    out.to_csv(output_path, index=False)
    print(f"Saved fresh test predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain", action="store_true",
        help="regenerate predictions_test.csv instead of reusing an existing file",
    )
    parser.add_argument(
        "--split", choices=["auto", "scaffold", "molecule"], default="auto",
        help="split used when generating predictions (default: scaffold when possible)",
    )
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="number of bootstrap replicates (default: 1000)")
    args = parser.parse_args()

    pred_path = "predictions_test.csv"
    if args.retrain or not os.path.exists(pred_path):
        if os.path.exists(pred_path) and args.retrain:
            print("--retrain supplied: existing predictions_test.csv will be overwritten.")
        make_predictions(pred_path, split_mode=args.split)
    else:
        print(
            "Using existing predictions_test.csv. Make sure it was generated from "
            "the current dataset and intended split. Use --retrain to regenerate it."
        )

    df = pd.read_csv(pred_path)
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"{pred_path} is missing required columns: {missing}")

    models = [c for c in MODEL_NAMES if c in df.columns]
    if not models:
        sys.exit(
            f"{pred_path} contains none of the required model columns: {MODEL_NAMES}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate numeric values and avoid silently evaluating malformed rows.
    numeric_cols = ["y_true"] + models
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols).copy()
    if len(df) < before:
        warnings.warn(f"Dropped {before - len(df)} rows with missing/non-finite labels or predictions.")
    if df.empty:
        sys.exit("No valid rows remain after checking labels and predictions.")

    # Avoid double-counting identical molecule-target rows in evaluation by default;
    # do not silently drop them because repeated assay measurements may be meaningful.
    duplicate_count = df.duplicated(["molecule_chembl_id", "target"]).sum()
    if duplicate_count:
        warnings.warn(
            f"Found {duplicate_count} duplicate molecule-target rows. They are retained; "
            "inspect repeated assays/records and document how they were handled."
        )

    df["Ensemble(mean)"] = df[models].mean(axis=1)
    all_models = models + ["Ensemble(mean)"]
    y_true = df["y_true"].to_numpy()
    groups = df["molecule_chembl_id"].astype(str).to_numpy()

    # Overall metrics and group-bootstrap confidence intervals.
    overall_rows = []
    for name in all_models:
        pred = df[name].to_numpy()
        metrics = score(y_true, pred)
        r2_ci, rmse_ci = bootstrap_ci(
            y_true, pred, groups=groups, n_boot=max(100, args.bootstrap), seed=42
        )
        metrics.update({
            "Model": name,
            "R2_CI": f"[{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]",
            "RMSE_CI": f"[{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]",
        })
        overall_rows.append(metrics)

    overall = pd.DataFrame(overall_rows)[
        ["Model", "N", "R2", "R2_CI", "RMSE", "RMSE_CI", "MAE",
         "Pearson_r", "Spearman_rho", "Within_0.5_log", "Within_1.0_log"]
    ]
    overall.to_csv("evaluation_overall.csv", index=False)
    print(f"\n===== Overall held-out set (n={len(df)}) =====")
    print(overall.round(3).to_string(index=False))

    # Per-target metrics, with safe handling for small/constant target subsets.
    per_target_rows = []
    for target, group in df.groupby("target", dropna=False):
        yt = group["y_true"].to_numpy()
        for name in all_models:
            metrics = score(yt, group[name].to_numpy())
            metrics.update(Model=name, Target=target, N=len(group))
            per_target_rows.append(metrics)

    per_target = pd.DataFrame(per_target_rows)[
        ["Model", "Target", "N", "R2", "RMSE", "MAE",
         "Pearson_r", "Spearman_rho", "Within_1.0_log"]
    ]
    per_target.to_csv("evaluation_per_target.csv", index=False)
    print("\n===== Per target =====")
    print(per_target.round(3).to_string(index=False))

    # Prediction-vs-actual plots.
    colors = {"her2": "tab:blue", "cdk4": "tab:orange"}
    pred_values = df[all_models].to_numpy(dtype=float)
    lo = min(float(y_true.min()), float(np.nanmin(pred_values))) - 0.3
    hi = max(float(y_true.max()), float(np.nanmax(pred_values))) + 0.3
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        lo, hi = -1.0, 1.0

    fig, axes = plt.subplots(1, len(all_models), figsize=(4.6 * len(all_models), 4.6),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    overall_indexed = overall.set_index("Model")
    for ax, name in zip(axes, all_models):
        for target, group in df.groupby("target"):
            ax.scatter(
                group["y_true"], group[name], s=12, alpha=0.6,
                color=colors.get(str(target).lower()), label=str(target).upper()
            )
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                        color="grey", alpha=0.12)
        row = overall_indexed.loc[name]
        ax.set_title(f"{name}\nR²={row['R2']:.2f}  RMSE={row['RMSE']:.2f}")
        ax.set_xlabel("Actual pIC50")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel("Predicted pIC50")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("eval_pred_vs_actual.png", dpi=200)
    plt.close()

    # Residual plot.
    fig, axes = plt.subplots(1, len(all_models), figsize=(4.6 * len(all_models), 3.8),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, all_models):
        residuals = df[name] - df["y_true"]
        ax.scatter(df[name], residuals, s=10, alpha=0.5)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(f"{name}\nmean residual={residuals.mean():+.2f}")
        ax.set_xlabel("Predicted pIC50")
    axes[0].set_ylabel("Residual (predicted − actual)")
    plt.tight_layout()
    plt.savefig("eval_residuals.png", dpi=200)
    plt.close()

    # Metric bars.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        ax.bar(overall["Model"], overall[metric], color="tab:blue")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig("eval_metric_bars.png", dpi=200)
    plt.close()

    print(
        "\nSaved evaluation_overall.csv, evaluation_per_target.csv, "
        "eval_pred_vs_actual.png, eval_residuals.png, eval_metric_bars.png"
    )


if __name__ == "__main__":
    main()
