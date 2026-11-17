"""
Exploratory HER2 vs CDK4 paired prediction analysis.

Input:
    pair_dataset.csv
Expected columns:
    molecule_chembl_id, canonical_smiles, protein_id, target, pIC50,
    bit_0 ... bit_2047, and protein features.

Method:
    - GroupKFold by molecule ID. A molecule is excluded from training in the
      fold where its paired predictions are generated, across BOTH targets.
    - Train RF, XGBoost, and SVR using the same feature matrix/pipeline
      structure as the supplied training code.
    - For each held-out molecule, predict against BOTH target-specific protein
      feature vectors by reusing its fingerprint bits and replacing protein
      features with the target's protein feature vector.
    - Model SD is reported as disagreement, NOT a calibrated prediction interval.
    - Similarity is max Morgan/Tanimoto similarity to training molecules for
      the relevant target in that fold.

Important:
    This creates exploratory, out-of-fold model preference scores. It does not
    establish biological selectivity. The supplied CSV has no assay metadata
    sufficient to verify cross-target assay comparability. Do not describe
    rankings as validated selectivity until assay endpoints/conditions are checked.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

DATA_PATH = Path("pair_dataset.csv")
OUT_DIR = Path("selectivity_results")
N_SPLITS = 5
RANDOM_STATE = 42
PREFERENCE_THRESHOLD = 0.5  # exploratory threshold in pIC50 units
TARGETS = ("her2", "cdk4")


def make_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_features="sqrt", n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5,
            tree_method="hist", random_state=RANDOM_STATE,
            n_jobs=-1, objective="reg:squarederror"
        ),
        "SVR": Pipeline([
            ("scale", ColumnTransformer(
                [("p", StandardScaler(), slice(2048, None))],
                remainder="passthrough"
            )),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
        ]),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    required = {"molecule_chembl_id", "canonical_smiles", "target", "pIC50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["target"] = df["target"].astype(str).str.lower().str.strip()
    df = df[df["target"].isin(TARGETS)].copy()
    df = df.dropna(subset=["molecule_chembl_id", "canonical_smiles", "pIC50"])

    bit_cols = sorted(
        [c for c in df.columns if c.startswith("bit_")],
        key=lambda c: int(c.split("_")[1])
    )
    if len(bit_cols) != 2048:
        raise ValueError(f"Expected 2048 bit_* columns, found {len(bit_cols)}")

    protein_cols = [c for c in df.columns
                    if c not in {"molecule_chembl_id", "canonical_smiles",
                                 "protein_id", "target", "pIC50"}
                    and c not in bit_cols]
    feature_cols = bit_cols + protein_cols
    if not protein_cols:
        raise ValueError("No protein feature columns found.")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["pIC50"].to_numpy(dtype=float)
    groups = df["molecule_chembl_id"].astype(str).to_numpy()

    # Basic source-data coverage and observed overlap.
    counts = df.groupby("target").agg(
        records=("molecule_chembl_id", "size"),
        unique_compounds=("molecule_chembl_id", "nunique"),
        pIC50_min=("pIC50", "min"),
        pIC50_median=("pIC50", "median"),
        pIC50_max=("pIC50", "max"),
    )
    per_target_ids = {
        t: set(df.loc[df.target == t, "molecule_chembl_id"].astype(str))
        for t in TARGETS
    }
    observed_overlap = per_target_ids["her2"] & per_target_ids["cdk4"]
    coverage = counts.reset_index()
    coverage["measured_compounds_in_both_targets"] = len(observed_overlap)
    coverage["assay_comparability"] = (
        "UNVERIFIED: input has no assay ID/type/condition metadata"
    )
    coverage.to_csv(OUT_DIR / "target_coverage_and_comparability.csv", index=False)

    # Target feature vectors must be unambiguous (one vector per target).
    target_features = {}
    for target in TARGETS:
        subset = df.loc[df.target == target, protein_cols]
        nunique = subset.nunique(dropna=False)
        if (nunique > 1).any():
            varying = nunique[nunique > 1].index.tolist()
            raise ValueError(
                f"Protein features vary within target {target}: {varying[:10]}. "
                "Confirm the feature construction before proceeding."
            )
        target_features[target] = subset.iloc[0].to_numpy(dtype=np.float32)

    # One fingerprint and canonical SMILES per unique molecule.
    mols = (df[["molecule_chembl_id", "canonical_smiles"] + bit_cols]
            .drop_duplicates("molecule_chembl_id")
            .set_index("molecule_chembl_id"))
    mol_ids = mols.index.astype(str).to_numpy()
    smiles_by_id = mols["canonical_smiles"].astype(str).to_dict()
    bits_by_id = {
        str(idx): row[bit_cols].to_numpy(dtype=np.float32)
        for idx, row in mols.iterrows()
    }

    # Build candidate features for each molecule x target. Feature order follows
    # the supplied training code: 2048 fingerprint bits, then protein features.
    candidate_X = {}
    for mid in mol_ids:
        for target in TARGETS:
            candidate_X[(mid, target)] = np.concatenate(
                [bits_by_id[mid], target_features[target]]
            ).astype(np.float32)

    # Morgan fingerprints for similarity. If RDKit is unavailable, analysis
    # continues but similarity is marked missing rather than fabricated.
    similarity_available = True
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        rdkit_mols = {}
        rdkit_fps = {}
        for mid, smi in smiles_by_id.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rdkit_mols[mid] = mol
                rdkit_fps[mid] = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048
                )
    except ImportError:
        similarity_available = False
        warnings.warn("RDKit not installed; chemical similarity will be blank.")

    splitter = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    prediction_rows = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups), start=1):
        train_df = df.iloc[tr_idx].copy()
        train_ids = set(train_df["molecule_chembl_id"].astype(str))
        heldout_ids = sorted(set(df.iloc[va_idx]["molecule_chembl_id"].astype(str)))
        models = make_models()
        for name, model in models.items():
            model.fit(X[tr_idx], y[tr_idx])

        for mid in heldout_ids:
            for target in TARGETS:
                x_new = candidate_X[(mid, target)].reshape(1, -1)
                preds = {
                    name: float(model.predict(x_new)[0])
                    for name, model in models.items()
                }
                values = np.array(list(preds.values()), dtype=float)
                sim = np.nan
                if similarity_available and mid in rdkit_fps:
                    target_train_ids = set(
                        train_df.loc[train_df.target == target,
                                     "molecule_chembl_id"].astype(str)
                    )
                    train_fps = [
                        rdkit_fps[tid] for tid in target_train_ids
                        if tid != mid and tid in rdkit_fps
                    ]
                    if train_fps:
                        sim = max(
                            DataStructs.TanimotoSimilarity(rdkit_fps[mid], fp)
                            for fp in train_fps
                        )
                prediction_rows.append({
                    "fold": fold,
                    "molecule_chembl_id": mid,
                    "canonical_smiles": smiles_by_id.get(mid, ""),
                    "target": target,
                    **{f"pred_{k}": v for k, v in preds.items()},
                    "pred_mean_pIC50": float(values.mean()),
                    "model_disagreement_sd": float(values.std(ddof=1)),
                    "max_tanimoto_to_target_training": sim,
                    "held_out_by_molecule": True,
                })

    pred = pd.DataFrame(prediction_rows)
    pred.to_csv(OUT_DIR / "paired_oof_predictions_long.csv", index=False)

    wide = pred.pivot(index="molecule_chembl_id", columns="target",
                      values="pred_mean_pIC50")
    wide.columns = [f"pred_mean_pIC50_{c}" for c in wide.columns]
    wide = wide.reset_index()
    pred_meta = pred.drop_duplicates("molecule_chembl_id")[
        ["molecule_chembl_id", "canonical_smiles", "fold"]
    ]
    wide = wide.merge(pred_meta, on="molecule_chembl_id", how="left")

    # Add per-target model disagreement and similarity.
    for target in TARGETS:
        sub = pred[pred.target == target][[
            "molecule_chembl_id", "model_disagreement_sd",
            "max_tanimoto_to_target_training"
        ]].rename(columns={
            "model_disagreement_sd": f"{target}_model_disagreement_sd",
            "max_tanimoto_to_target_training": f"{target}_max_tanimoto"
        })
        wide = wide.merge(sub, on="molecule_chembl_id", how="left")

    hcol, ccol = "pred_mean_pIC50_her2", "pred_mean_pIC50_cdk4"
    if hcol not in wide or ccol not in wide:
        raise RuntimeError("Could not construct paired predictions for both targets.")
    wide["delta_pIC50_HER2_minus_CDK4"] = wide[hcol] - wide[ccol]
    wide["predicted_potency_ratio_HER2_over_CDK4"] = (
        10.0 ** wide["delta_pIC50_HER2_minus_CDK4"].clip(-12, 12)
    )
    wide["exploratory_preference"] = np.select(
        [
            wide["delta_pIC50_HER2_minus_CDK4"] >= PREFERENCE_THRESHOLD,
            wide["delta_pIC50_HER2_minus_CDK4"] <= -PREFERENCE_THRESHOLD,
        ],
        ["HER2-preferred prediction", "CDK4-preferred prediction"],
        default="No strong predicted preference"
    )
    wide["ranking_status"] = (
        "EXPLORATORY MODEL PREFERENCE ONLY; assay comparability not verified"
    )
    wide = wide.sort_values("delta_pIC50_HER2_minus_CDK4", ascending=False)
    wide["rank_HER2_preference"] = np.arange(1, len(wide) + 1)
    wide.to_csv(OUT_DIR / "exploratory_selectivity_rankings.csv", index=False)

    # Observed paired labels are a separate small check, not a model ranking.
    observed = df.pivot_table(
        index=["molecule_chembl_id", "canonical_smiles"],
        columns="target", values="pIC50", aggfunc="median"
    ).reset_index()
    if "her2" in observed.columns and "cdk4" in observed.columns:
        paired = observed.dropna(subset=["her2", "cdk4"]).copy()
        paired["observed_delta_pIC50_HER2_minus_CDK4"] = paired["her2"] - paired["cdk4"]
        paired.to_csv(OUT_DIR / "observed_paired_compounds.csv", index=False)

    # Scatter plot of out-of-fold paired model predictions.
    plt.figure(figsize=(7, 6))
    plt.scatter(wide[ccol], wide[hcol], alpha=0.45, s=16)
    lo = float(min(wide[ccol].min(), wide[hcol].min()))
    hi = float(max(wide[ccol].max(), wide[hcol].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Predicted CDK4 pIC50 (OOF)")
    plt.ylabel("Predicted HER2 pIC50 (OOF)")
    plt.title("Exploratory paired out-of-fold predictions")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "paired_prediction_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(wide["delta_pIC50_HER2_minus_CDK4"].dropna(), bins=35)
    plt.axvline(PREFERENCE_THRESHOLD, linestyle="--")
    plt.axvline(-PREFERENCE_THRESHOLD, linestyle="--")
    plt.xlabel("Predicted ΔpIC50 (HER2 − CDK4)")
    plt.ylabel("Number of compounds")
    plt.title("Exploratory predicted target preference")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "delta_pIC50_distribution.png", dpi=180)
    plt.close()

    # Report standard metrics on observed held-out rows for each target and model.
    # These metrics are OOF and grouped by molecule, matching the paired workflow.
    metric_rows = []
    for target in TARGETS:
        obs = df[df.target == target][["molecule_chembl_id", "pIC50"]].copy()
        sub = pred[pred.target == target]
        joined = obs.merge(sub, on="molecule_chembl_id", how="inner")
        for model_name in ["RandomForest", "XGBoost", "SVR"]:
            col = f"pred_{model_name}"
            metric_rows.append({
                "target": target,
                "model": model_name,
                "n": len(joined),
                "R2": r2_score(joined["pIC50"], joined[col]),
                "RMSE": mean_squared_error(joined["pIC50"], joined[col]) ** 0.5,
                "MAE": mean_absolute_error(joined["pIC50"], joined[col]),
            })
        joined["pred_ensemble_mean"] = joined[
            ["pred_RandomForest", "pred_XGBoost", "pred_SVR"]
        ].mean(axis=1)
        metric_rows.append({
            "target": target, "model": "Ensemble(mean)", "n": len(joined),
            "R2": r2_score(joined["pIC50"], joined["pred_ensemble_mean"]),
            "RMSE": mean_squared_error(joined["pIC50"], joined["pred_ensemble_mean"]) ** 0.5,
            "MAE": mean_absolute_error(joined["pIC50"], joined["pred_ensemble_mean"]),
        })
    pd.DataFrame(metric_rows).to_csv(
        OUT_DIR / "grouped_oof_metrics.csv", index=False
    )

    print("\nTarget coverage:")
    print(coverage.to_string(index=False))
    print(f"\nMeasured compounds present for both targets: {len(observed_overlap)}")
    print("Assay comparability: UNVERIFIED from available columns.")
    print(f"Saved results to: {OUT_DIR.resolve()}")
    print("Do not describe exploratory rankings as validated biological selectivity.")


if __name__ == "__main__":
    main()
