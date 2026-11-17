"""
HER2 vs CDK4 selectivity analysis.

This script creates a compound-level table with predictions for BOTH targets,
computes Delta pIC50 = predicted pIC50(HER2) - predicted pIC50(CDK4), ranks
target preference, calculates chemical similarity to each target's training
compounds when source SMILES are available, and adds uncertainty estimates
when an ensemble of saved models is available.

IMPORTANT:
- This is a computational prioritization tool, not experimental proof of selectivity.
- HER2 and CDK4 values should only be compared if assay endpoints, units,
  transformations, and data provenance are sufficiently comparable.
- This script does not invent missing predictions. A compound missing a valid
  prediction for either target is not assigned a Delta pIC50 or rank.
- Uncertainty from model disagreement is a rough diagnostic, not a calibrated
  prediction interval.
- Edit the configuration below to match the exact filenames and feature layout
  used by your trained models.

Recommended input:
  pair_dataset.csv with molecule_chembl_id, target, SMILES, pIC50
  X_pairs.npy and y_pIC50.npy aligned row-for-row with pair_dataset.csv
  dual_models.joblib OR final_models.joblib (if these actually contain models
  trained for both HER2 and CDK4, with a compatible feature pipeline)

Alternative:
  predictions_test.csv containing molecule_chembl_id, target, y_true, and one
  or more prediction columns can be used for observed held-out predictions.
  However, a test prediction file generally won't contain predictions for the
  same compounds against both targets; only matched pairs are compared.

Install:
  pip install pandas numpy scikit-learn matplotlib joblib
  conda install -c conda-forge rdkit  # optional, for chemical similarity/scaffolds

Run:
  python selectivity_analysis.py

The script first tries to use prediction columns already present in
predictions_all_targets.csv. If absent, it tries to load a saved model artifact
and feature matrices. It will stop with an informative error rather than
fabricate predictions if the artifact schema is not compatible.
"""

from pathlib import Path
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- CONFIGURATION -------------------------
DATASET_FILE = Path("pair_dataset.csv")
PREDICTIONS_FILE = Path("predictions_all_targets.csv")
MODEL_CANDIDATES = [Path("dual_models.joblib"), Path("final_models.joblib")]
FEATURES_FILE = Path("X_pairs.npy")
LABELS_FILE = Path("y_pIC50.npy")

ID_COL = "molecule_chembl_id"
TARGET_COL = "target"
# Add/modify names here to match pair_dataset.csv.
SMILES_CANDIDATES = ["canonical_smiles", "smiles", "SMILES", "canonical_SMILES"]
PIC50_CANDIDATES = ["pIC50", "pIC50_value", "y_pIC50", "pIC50_standard_value"]

HER2_NAMES = {"her2", "erbb2", "p04626"}
CDK4_NAMES = {"cdk4", "cdk-4", "p11802"}
# Minimum test-independent training compounds per target to issue rankings.
MIN_COMPOUNDS_PER_TARGET = 30
# Treat Delta pIC50 magnitudes below this as no clear predicted preference.
PREFERENCE_THRESHOLD = 0.5
# Similarity warning threshold: tune based on validation, 0.4 is only a heuristic.
LOW_SIMILARITY_THRESHOLD = 0.4
# ----------------------------------------------------------------


def normalize_target(value):
    value = str(value).strip().lower()
    if value in HER2_NAMES:
        return "HER2"
    if value in CDK4_NAMES:
        return "CDK4"
    return None


def first_existing_column(frame, candidates):
    return next((col for col in candidates if col in frame.columns), None)


def find_smiles_column(frame):
    return first_existing_column(frame, SMILES_CANDIDATES)


def load_data():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATASET_FILE}. Run this script from your MINI_PROJECT folder."
        )
    meta = pd.read_csv(DATASET_FILE)
    required = {ID_COL, TARGET_COL}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{DATASET_FILE} is missing required columns: {sorted(missing)}")
    meta = meta.copy()
    meta["target_norm"] = meta[TARGET_COL].map(normalize_target)
    meta = meta[meta["target_norm"].notna()].copy()
    if meta.empty:
        raise ValueError(
            f"No HER2/CDK4 target labels found in '{TARGET_COL}'. "
            "Inspect target values and update normalize_target()."
        )
    meta[ID_COL] = meta[ID_COL].astype(str)
    smiles_col = find_smiles_column(meta)
    if smiles_col:
        meta["analysis_smiles"] = meta[smiles_col].fillna("").astype(str)
    else:
        meta["analysis_smiles"] = ""
        warnings.warn(
            f"No recognized SMILES column found. Chemical similarity will be unavailable. "
            f"Columns found: {list(meta.columns)}"
        )
    pic50_col = first_existing_column(meta, PIC50_CANDIDATES)
    if pic50_col:
        meta["observed_pIC50"] = pd.to_numeric(meta[pic50_col], errors="coerce")
    else:
        meta["observed_pIC50"] = np.nan
        warnings.warn(
            "No pIC50 label column found in pair_dataset.csv. "
            "Assay comparability and observed coverage cannot be assessed automatically."
        )
    return meta


def read_prediction_table():
    if not PREDICTIONS_FILE.exists():
        return None
    pred = pd.read_csv(PREDICTIONS_FILE)
    required = {ID_COL, TARGET_COL}
    if not required.issubset(pred.columns):
        warnings.warn(f"{PREDICTIONS_FILE} lacks {required}; ignoring it.")
        return None
    pred = pred.copy()
    pred["target_norm"] = pred[TARGET_COL].map(normalize_target)
    pred = pred[pred["target_norm"].notna()].copy()
    # Accept common prediction column names. Never treat y_true as a prediction.
    candidate_cols = [
        c for c in ["predicted_pIC50", "prediction", "y_pred",
                    "Ensemble(mean)", "RandomForest", "XGBoost", "SVR"]
        if c in pred.columns
    ]
    if not candidate_cols:
        warnings.warn(f"{PREDICTIONS_FILE} has no recognized prediction column.")
        return None
    # Prefer ensemble; otherwise average available prediction columns.
    if "Ensemble(mean)" in candidate_cols:
        pred["predicted_pIC50"] = pd.to_numeric(pred["Ensemble(mean)"], errors="coerce")
    elif "predicted_pIC50" in candidate_cols:
        pred["predicted_pIC50"] = pd.to_numeric(pred["predicted_pIC50"], errors="coerce")
    elif "prediction" in candidate_cols:
        pred["predicted_pIC50"] = pd.to_numeric(pred["prediction"], errors="coerce")
    elif "y_pred" in candidate_cols:
        pred["predicted_pIC50"] = pd.to_numeric(pred["y_pred"], errors="coerce")
    else:
        pred["predicted_pIC50"] = pred[candidate_cols].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
    model_cols = [c for c in ["RandomForest", "XGBoost", "SVR"] if c in pred.columns]
    if len(model_cols) >= 2:
        pred["model_disagreement_sd"] = pred[model_cols].std(axis=1, ddof=1)
    else:
        pred["model_disagreement_sd"] = np.nan
    return pred[[ID_COL, "target_norm", "predicted_pIC50", "model_disagreement_sd"]]


def predict_from_artifact(meta):
    """
    Attempt inference only if the artifact structure is recognized.

    Supported structures:
      {"HER2": fitted_estimator, "CDK4": fitted_estimator}
      {"her2": fitted_estimator, "cdk4": fitted_estimator}
      {"models": {"HER2": fitted_estimator, "CDK4": fitted_estimator}}
      {"HER2": {"SVR": fitted_estimator, ...}, "CDK4": {...}}
      {"models": {"HER2": {"SVR": estimator, ...}, "CDK4": {...}}}

    Each estimator must accept the matching row from X_pairs.npy. If your models
    were trained on target-specific feature matrices (X_her2.npy/X_cdk4.npy),
    edit this function to use those exact matrices and their aligned IDs.
    """
    import joblib

    artifact_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if artifact_path is None:
        return None, None

    artifact = joblib.load(artifact_path)
    if not FEATURES_FILE.exists():
        warnings.warn(f"{FEATURES_FILE} not found; cannot predict from {artifact_path}.")
        return None, None
    X = np.load(FEATURES_FILE)
    if len(X) != len(meta):
        warnings.warn(
            f"X rows ({len(X)}) do not match pair_dataset rows ({len(meta)}); "
            "cannot safely align predictions. Use the exact feature matrix and metadata."
        )
        return None, None

    container = artifact.get("models", artifact) if isinstance(artifact, dict) else None
    if not isinstance(container, dict):
        warnings.warn(f"Unsupported model artifact structure in {artifact_path}.")
        return None, None

    outputs = []
    for target in ["HER2", "CDK4"]:
        model_obj = None
        for key, value in container.items():
            if normalize_target(key) == target:
                model_obj = value
                break
        if model_obj is None:
            warnings.warn(f"{artifact_path} has no recognized model entry for {target}.")
            return None, None
        # If target entry contains multiple estimators, average their predictions.
        if isinstance(model_obj, dict):
            estimators = [v for v in model_obj.values() if hasattr(v, "predict")]
            if not estimators:
                warnings.warn(f"No fitted estimators found for target {target}.")
                return None, None
        elif hasattr(model_obj, "predict"):
            estimators = [model_obj]
        else:
            warnings.warn(f"Unsupported estimator object for target {target}.")
            return None, None

        target_mask = meta["target_norm"].eq(target).to_numpy()
        if not target_mask.any():
            continue
        target_X = X[target_mask]
        individual = np.column_stack([est.predict(target_X) for est in estimators])
        outputs.append(pd.DataFrame({
            ID_COL: meta.loc[target_mask, ID_COL].to_numpy(),
            "target_norm": target,
            "predicted_pIC50": individual.mean(axis=1),
            "model_disagreement_sd": (
                individual.std(axis=1, ddof=1) if individual.shape[1] >= 2
                else np.full(individual.shape[0], np.nan)
            ),
        }))
    if not outputs:
        return None, None
    return pd.concat(outputs, ignore_index=True), str(artifact_path)


def compute_similarity(meta):
    """Max Morgan-fingerprint Tanimoto similarity to observed compounds per target."""
    if not meta["analysis_smiles"].astype(bool).any():
        return None
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        warnings.warn("RDKit not installed; similarity will be left blank.")
        return None

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    valid = {}
    for _, row in meta.iterrows():
        smi = row["analysis_smiles"]
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            valid[row[ID_COL]] = (mol, fpgen.GetFingerprint(mol))

    similarity = {}
    for target in ["HER2", "CDK4"]:
        train_ids = set(meta.loc[meta["target_norm"].eq(target), ID_COL])
        target_fps = [valid[i][1] for i in train_ids if i in valid]
        for mol_id in meta[ID_COL].unique():
            if mol_id not in valid or not target_fps:
                similarity[(mol_id, target)] = np.nan
                continue
            fp = valid[mol_id][1]
            similarity[(mol_id, target)] = max(
                DataStructs.TanimotoSimilarity(fp, train_fp) for train_fp in target_fps
            )
    return similarity


def main():
    meta = load_data()
    preds = read_prediction_table()
    prediction_source = str(PREDICTIONS_FILE)

    # Existing prediction tables are preferred only if they actually include
    # matched predictions. Otherwise try the saved dual-target model artifact.
    if preds is None:
        preds, artifact_name = predict_from_artifact(meta)
        if preds is not None:
            prediction_source = artifact_name

    if preds is None:
        raise RuntimeError(
            "Could not obtain predictions for both targets. This script will not "
            "fabricate them. Create predictions_all_targets.csv with columns "
            "molecule_chembl_id,target,predicted_pIC50 (one row per compound-target), "
            "or adapt predict_from_artifact() to the actual structure of your saved models."
        )

    preds["target_norm"] = preds["target_norm"].map(normalize_target)
    preds = preds[preds["target_norm"].isin(["HER2", "CDK4"])].copy()
    preds["predicted_pIC50"] = pd.to_numeric(preds["predicted_pIC50"], errors="coerce")
    preds = preds.replace([np.inf, -np.inf], np.nan)
    preds = preds.dropna(subset=[ID_COL, "target_norm", "predicted_pIC50"])
    # One prediction per compound-target is required. Average duplicate predictions,
    # and preserve ensemble disagreement as the mean within-group SD when possible.
    preds = preds.groupby([ID_COL, "target_norm"], as_index=False).agg(
        predicted_pIC50=("predicted_pIC50", "mean"),
        model_disagreement_sd=("model_disagreement_sd", "mean"),
    )

    wide = preds.pivot(index=ID_COL, columns="target_norm", values="predicted_pIC50")
    wide.columns.name = None
    wide = wide.rename(columns={"HER2": "pred_pIC50_HER2", "CDK4": "pred_pIC50_CDK4"})
    wide = wide.reset_index()
    if not {"pred_pIC50_HER2", "pred_pIC50_CDK4"}.issubset(wide.columns):
        raise RuntimeError(
            "Predictions do not contain both HER2 and CDK4. Generate predictions for "
            "the same compounds against both targets before calculating selectivity."
        )

    uncertainty = preds.pivot(
        index=ID_COL, columns="target_norm", values="model_disagreement_sd"
    )
    uncertainty.columns.name = None
    uncertainty = uncertainty.rename(
        columns={"HER2": "model_disagreement_sd_HER2", "CDK4": "model_disagreement_sd_CDK4"}
    ).reset_index()
    wide = wide.merge(uncertainty, on=ID_COL, how="left")
    wide["delta_pIC50_HER2_minus_CDK4"] = wide["pred_pIC50_HER2"] - wide["pred_pIC50_CDK4"]
    wide["predicted_fold_difference"] = 10 ** np.clip(
        wide["delta_pIC50_HER2_minus_CDK4"], -300, 300
    )
    wide["predicted_preference"] = np.select(
        [
            wide["delta_pIC50_HER2_minus_CDK4"] >= PREFERENCE_THRESHOLD,
            wide["delta_pIC50_HER2_minus_CDK4"] <= -PREFERENCE_THRESHOLD,
        ],
        ["HER2-preferred", "CDK4-preferred"],
        default="No clear preference (< threshold)",
    )

    # Attach metadata, but avoid accidentally duplicating one compound across
    # multiple assay rows by first creating a compound-level metadata table.
    meta_cols = [ID_COL, "analysis_smiles"]
    compound_meta = meta[meta_cols].drop_duplicates(ID_COL, keep="first")
    result = wide.merge(compound_meta, on=ID_COL, how="left")

    # Chemical similarity is max Tanimoto similarity to compounds with records for
    # each target in this dataset. This is a descriptive proxy, not a true train-only
    # applicability-domain estimate unless the metadata itself is training-only.
    similarity = compute_similarity(meta)
    if similarity is not None:
        result["max_similarity_HER2"] = result[ID_COL].map(
            lambda x: similarity.get((x, "HER2"), np.nan)
        )
        result["max_similarity_CDK4"] = result[ID_COL].map(
            lambda x: similarity.get((x, "CDK4"), np.nan)
        )
        result["low_similarity_warning"] = (
            result[["max_similarity_HER2", "max_similarity_CDK4"]].min(axis=1)
            < LOW_SIMILARITY_THRESHOLD
        )
    else:
        result["max_similarity_HER2"] = np.nan
        result["max_similarity_CDK4"] = np.nan
        result["low_similarity_warning"] = "Similarity unavailable"

    # Observed data coverage and basic label range check.
    coverage = meta.groupby("target_norm").agg(
        n_rows=("target_norm", "size"),
        n_compounds=(ID_COL, "nunique"),
        n_observed_pIC50=("observed_pIC50", "count"),
        observed_pIC50_min=("observed_pIC50", "min"),
        observed_pIC50_max=("observed_pIC50", "max"),
    ).reset_index()
    coverage.to_csv("selectivity_target_coverage.csv", index=False)
    coverage_by_target = coverage.set_index("target_norm")
    adequate_coverage = all(
        t in coverage_by_target.index
        and coverage_by_target.loc[t, "n_compounds"] >= MIN_COMPOUNDS_PER_TARGET
        for t in ["HER2", "CDK4"]
    )
    result["ranking_eligible"] = adequate_coverage
    if adequate_coverage:
        result = result.sort_values("delta_pIC50_HER2_minus_CDK4", ascending=False)
        result["rank_HER2_preference"] = np.arange(1, len(result) + 1)
        result["rank_CDK4_preference"] = result["delta_pIC50_HER2_minus_CDK4"].rank(
            ascending=True, method="min"
        ).astype(int)
    else:
        result["rank_HER2_preference"] = np.nan
        result["rank_CDK4_preference"] = np.nan
        warnings.warn(
            "Insufficient observed compound coverage for at least one target "
            f"(minimum configured: {MIN_COMPOUNDS_PER_TARGET} unique compounds). "
            "Ranks are withheld. Check selectivity_target_coverage.csv."
        )

    # Write full table and two ranked shortlists.
    columns = [
        ID_COL, "analysis_smiles", "pred_pIC50_HER2", "pred_pIC50_CDK4",
        "delta_pIC50_HER2_minus_CDK4", "predicted_fold_difference",
        "predicted_preference", "model_disagreement_sd_HER2",
        "model_disagreement_sd_CDK4", "max_similarity_HER2", "max_similarity_CDK4",
        "low_similarity_warning", "ranking_eligible",
        "rank_HER2_preference", "rank_CDK4_preference",
    ]
    result[columns].to_csv("selectivity_rankings.csv", index=False)
    result.sort_values("delta_pIC50_HER2_minus_CDK4", ascending=False).head(25).to_csv(
        "top_25_HER2_preferred.csv", index=False
    )
    result.sort_values("delta_pIC50_HER2_minus_CDK4", ascending=True).head(25).to_csv(
        "top_25_CDK4_preferred.csv", index=False
    )

    # Plot matched-target prediction landscape.
    plot_df = result.dropna(subset=["pred_pIC50_HER2", "pred_pIC50_CDK4"])
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            plot_df["pred_pIC50_CDK4"], plot_df["pred_pIC50_HER2"],
            c=plot_df["delta_pIC50_HER2_minus_CDK4"], cmap="coolwarm",
            s=28, alpha=0.8, edgecolors="none"
        )
        low = min(plot_df["pred_pIC50_CDK4"].min(), plot_df["pred_pIC50_HER2"].min())
        high = max(plot_df["pred_pIC50_CDK4"].max(), plot_df["pred_pIC50_HER2"].max())
        ax.plot([low, high], [low, high], "k--", lw=1, label="Equal predicted pIC50")
        ax.set_xlabel("Predicted CDK4 pIC50")
        ax.set_ylabel("Predicted HER2 pIC50")
        ax.set_title("Predicted HER2 vs CDK4 activity")
        fig.colorbar(sc, ax=ax, label="ΔpIC50 (HER2 − CDK4)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("selectivity_landscape.png", dpi=200)
        plt.close(fig)

    print(f"Predictions source: {prediction_source}")
    print("\nTarget coverage:")
    print(coverage.to_string(index=False))
    print(f"\nMatched compounds with both predictions: {len(result)}")
    print("Saved: selectivity_rankings.csv, top_25_HER2_preferred.csv, "
          "top_25_CDK4_preferred.csv, selectivity_target_coverage.csv, "
          "selectivity_landscape.png (when plot data are available)")
    print(
        "\nInterpretation caution: these are model-based rankings. Compare targets "
        "only when assay endpoints and data processing are comparable. The similarity "
        "values are not train-only unless the input metadata is training-only, and "
        "model disagreement is not a calibrated prediction interval."
    )


if __name__ == "__main__":
    main()
