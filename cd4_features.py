"""
CD4 protein feature extraction: AAC + DPC + physicochemical.

Reuses the functions in erbb2_features.py (keep both files in the same folder).
CD4 = human T-cell surface glycoprotein CD4, UniProt P01730 (458 aa).

Install:  pip install requests biopython pandas numpy
"""

import os

import pandas as pd

from erbb2_features import (
    get_uniprot_sequence,
    protein_feature_vector,
    build_pair_matrix,
)

ACCESSION = "P01730"  # human CD4

if __name__ == "__main__":
    # --- 1) sequence ---
    seq = get_uniprot_sequence(ACCESSION, fallback_fasta=f"{ACCESSION}.fasta")

    # --- 2) ONE CD4 feature vector (AAC 20 + DPC 400 + physicochemical) ---
    vec = protein_feature_vector(seq)
    print(f"Feature vector length: {len(vec)} "
          f"(AAC=20, DPC=400, physchem={len(vec) - 420})")
    vec.to_frame("CD4").to_csv("cd4_feature_vector.csv")

    # --- 3) repeat for all 1,000 drug-protein pairs ---
    if os.path.exists("drug_protein_pairs.csv"):
        pairs = pd.read_csv("drug_protein_pairs.csv")
    else:  # placeholder so the script runs end to end
        pairs = pd.DataFrame({"drug_id": [f"drug_{i}" for i in range(1000)],
                              "protein_id": ACCESSION})

    X = build_pair_matrix(pairs, vec)
    X.to_csv("pair_features_cd4.csv", index=False)
    print("Pair feature matrix:", X.shape)
