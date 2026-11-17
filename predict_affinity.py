"""
Final output: predicted binding affinity (pIC50 / IC50) for a drug-protein pair.

Usage (run in the MINI_PROJECT folder):
  python predict_affinity.py --smiles "CCOc1cc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1" --protein P04626
  python predict_affinity.py --smiles "<SMILES>" --protein cdk4
  python predict_affinity.py --csv new_pairs.csv --out predictions.csv   # columns: smiles, protein
  python predict_affinity.py                                            # runs a built-in demo

--protein: UniProt accession (P04626 = HER2/ERBB2, P11802 = CDK4) or the alias her2 / cdk4.
The first run trains RF + XGBoost + SVR on ALL your data and saves final_models.joblib.

Needs: X_pairs.npy, y_pIC50.npy, pair_dataset.csv (from concat_drug_protein.py),
       erbb2_features.py (only for proteins other than HER2/CDK4), rdkit.
Install: pip install rdkit scikit-learn xgboost joblib pandas numpy requests biopython
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

N_BITS = 2048
MODEL_FILE = "final_models.joblib"
ALIASES = {"her2": "P04626", "erbb2": "P04626", "cdk4": "P11802"}
TEST_RMSE = 0.65   # typical held-out RMSE (pIC50 units) from the evaluation step

# Same fingerprint settings as the training data (verified: Morgan, radius 2, 2048 bits)
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)


# ---------------------------------------------------------------- training data
X = np.load("X_pairs.npy")
y = np.load("y_pIC50.npy")
meta = pd.read_csv("pair_dataset.csv", usecols=["protein_id", "target"])
assert len(meta) == len(X) == len(y)

# protein feature vector for each protein seen in training (identical to training rows)
protein_vecs = {pid: X[np.where(meta["protein_id"].to_numpy() == pid)[0][0], N_BITS:]
                for pid in meta["protein_id"].unique()}


def get_models():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    print("Training final models on all data (one-off, ~1 min) ...")
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                              n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.5,
                                tree_method="hist", random_state=42),
        "SVR": Pipeline([
            ("scale", ColumnTransformer([("p", StandardScaler(), slice(N_BITS, None))],
                                        remainder="passthrough")),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
        ]),
    }
    for m in models.values():
        m.fit(X, y)
    joblib.dump(models, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")
    return models


def get_protein_vec(protein):
    acc = ALIASES.get(str(protein).lower(), str(protein).upper())
    if acc in protein_vecs:
        return acc, protein_vecs[acc], True
    from erbb2_features import get_uniprot_sequence, protein_feature_vector
    seq = get_uniprot_sequence(acc, fallback_fasta=f"{acc}.fasta")
    vec = protein_feature_vector(seq).to_numpy(dtype=np.float32)
    protein_vecs[acc] = vec
    return acc, vec, False


def drug_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return _gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def nearest_training_similarity(fp, acc):
    """Max Tanimoto similarity to training drugs for this protein (applicability check)."""
    rows = np.where(meta["protein_id"].to_numpy() == acc)[0]
    if len(rows) == 0:
        return np.nan
    T = X[rows, :N_BITS] > 0
    q = fp > 0
    inter = (T & q).sum(axis=1)
    union = T.sum(axis=1) + q.sum() - inter
    return float((inter / np.maximum(union, 1)).max())


def predict(models, smiles, protein):
    acc, pvec, known = get_protein_vec(protein)
    fp = drug_fingerprint(smiles)
    x = np.hstack([fp, pvec]).reshape(1, -1)
    per_model = {n: float(m.predict(x)[0]) for n, m in models.items()}
    p = float(np.mean(list(per_model.values())))
    sim = nearest_training_similarity(fp, acc)

    if not known:
        note = "protein NOT in training data - prediction unreliable"
    elif sim < 0.3:
        note = "drug is structurally far from training compounds - low confidence"
    elif sim < 0.5:
        note = "moderate similarity to training compounds"
    else:
        note = "drug is similar to training compounds - higher confidence"
    strength = ("strong (IC50 < 100 nM)" if p >= 7 else
                "moderate (100 nM - 1 uM)" if p >= 6 else "weak (IC50 > 1 uM)")
    return {
        "protein": acc, "smiles": smiles,
        "pIC50_pred": round(p, 2),
        "IC50_nM_pred": round(10 ** (9 - p), 2),
        "approx_range_pIC50": f"{p - TEST_RMSE:.2f} to {p + TEST_RMSE:.2f}",
        "affinity_class": strength,
        "RF": round(per_model["RandomForest"], 2),
        "XGB": round(per_model["XGBoost"], 2),
        "SVR": round(per_model["SVR"], 2),
        "max_tanimoto_to_train": None if np.isnan(sim) else round(sim, 2),
        "confidence_note": note,
    }


def show(r):
    print("\n" + "=" * 62)
    print(f" Protein : {r['protein']}")
    print(f" Drug    : {r['smiles']}")
    print("-" * 62)
    print(f" Predicted pIC50        : {r['pIC50_pred']}   (range {r['approx_range_pIC50']})")
    print(f" Predicted IC50         : {r['IC50_nM_pred']} nM  -> {r['affinity_class']}")
    print(f" Individual models      : RF {r['RF']} | XGB {r['XGB']} | SVR {r['SVR']}")
    print(f" Similarity to training : {r['max_tanimoto_to_train']}  ({r['confidence_note']})")
    print("=" * 62)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles")
    ap.add_argument("--protein", default="P04626")
    ap.add_argument("--csv", help="CSV with columns smiles, protein")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    models = get_models()
    if a.csv:
        df = pd.read_csv(a.csv)
        res = pd.DataFrame([predict(models, r.smiles, r.protein) for r in df.itertuples()])
        res.to_csv(a.out, index=False)
        print(res[["protein", "pIC50_pred", "IC50_nM_pred", "affinity_class"]].to_string(index=False))
        print(f"\nSaved {a.out}")
    else:
        # default demo: lapatinib vs HER2 when no SMILES is given
        smi = a.smiles or "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1"
        show(predict(models, smi, a.protein))
