"""
Concatenate drug (Morgan fingerprint) + protein (AAC + DPC + physicochemical)
features for the HER2 and CDK4 datasets in this folder.

Inputs (already in this folder):
  her2_fingerprints.csv   cdk4_fingerprints.csv
  (each has molecule_chembl_id, canonical_smiles, median_ic50_nM, bit_0 ... bit_2047)
Also needs erbb2_features.py in this folder (protein feature functions).

Output:
  X_pairs.npy  (2735 x 2048+443)  y_pIC50.npy  pair_dataset.csv
Each row = [ drug fingerprint (2048) | protein feature vector (443) ].
"""

import numpy as np
import pandas as pd

from erbb2_features import get_uniprot_sequence, protein_feature_vector

# target name -> UniProt accession
TARGETS = {
    "her2": "P04626",   # ERBB2 / HER2 (ChEMBL CHEMBL1824)
    "cdk4": "P11802",   # CDK4        (ChEMBL CHEMBL331)
}

# ---- 1. protein feature vector for each target (computed once each) ----
prot_vec = {}
for name, acc in TARGETS.items():
    seq = get_uniprot_sequence(acc, fallback_fasta=f"{acc}.fasta")
    prot_vec[name] = protein_feature_vector(seq)
    print(f"{name} ({acc}): protein vector = {len(prot_vec[name])} features")
prot_cols = list(prot_vec["her2"].index)

# ---- 2. per target: load drug features, repeat the protein vector, hstack ----
X_parts, y_parts, meta_parts = [], [], []
for name, acc in TARGETS.items():
    # Read everything from ONE file so ids, bits and IC50 can never get out of step
    fp = pd.read_csv(f"{name}_fingerprints.csv")
    bit_cols = [c for c in fp.columns if c.startswith("bit_")]
    fp = fp.dropna(subset=["median_ic50_nM"]).drop_duplicates("molecule_chembl_id")
    fp = fp[fp["median_ic50_nM"] > 0].reset_index(drop=True)
    X_drug = fp[bit_cols].to_numpy(dtype=np.float32)               # (n, 2048)
    y = 9 - np.log10(fp["median_ic50_nM"].to_numpy())              # pIC50
    ids = fp[["molecule_chembl_id", "canonical_smiles"]]
    print(f"{name}: {len(fp)} molecules, {len(bit_cols)} fingerprint bits")

    X_prot = np.tile(prot_vec[name].to_numpy(dtype=np.float32), (len(X_drug), 1))
    X_parts.append(np.hstack([X_drug, X_prot]))                    # (n, 2491)
    y_parts.append(y)
    meta_parts.append(pd.DataFrame({
        "molecule_chembl_id": ids["molecule_chembl_id"],
        "canonical_smiles": ids["canonical_smiles"],
        "protein_id": acc,
        "target": name,
        "pIC50": y,
    }))
    print(f"{name}: {X_drug.shape} + {X_prot.shape} -> {X_parts[-1].shape}")

# ---- 3. stack both targets into one dataset ----
X = np.vstack(X_parts)
y = np.concatenate(y_parts)
meta = pd.concat(meta_parts, ignore_index=True)
print("Final X:", X.shape, "| y:", y.shape)

# ---- 4. save ----
np.save("X_pairs.npy", X)
np.save("y_pIC50.npy", y)
cols = [f"bit_{i}" for i in range(2048)] + prot_cols
pd.concat([meta, pd.DataFrame(X, columns=cols)], axis=1).to_csv("pair_dataset.csv", index=False)
print("Saved X_pairs.npy, y_pIC50.npy, pair_dataset.csv")
