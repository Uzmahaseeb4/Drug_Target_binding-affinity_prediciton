"""
ERBB2 (HER2) protein feature extraction: AAC + DPC + physicochemical.

Steps
1. Download the ERBB2 sequence from UniProt (accession P04626).
2. Build ONE feature vector = AAC (20) + DPC (400) + physicochemical (~25).
3. Repeat that vector for all 1,000 drug-protein pairs (same protein, so the
   protein block is identical in every row; drug features get concatenated later).

Install:  pip install requests biopython pandas numpy
"""

import io
import itertools
import os

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = ["".join(p) for p in itertools.product(AA, repeat=2)]  # 400


# ----------------------------------------------------------------------
# 1. Get the sequence
# ----------------------------------------------------------------------
def get_uniprot_sequence(accession: str = "P04626", fallback_fasta: str = None) -> str:
    """Fetch a protein sequence from UniProt REST. P04626 = human ERBB2."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        record = next(SeqIO.parse(io.StringIO(r.text), "fasta"))
        print(f"Fetched {record.id} | length = {len(record.seq)}")
        return str(record.seq)
    except Exception as e:
        if fallback_fasta and os.path.exists(fallback_fasta):
            print(f"UniProt fetch failed ({e}); reading {fallback_fasta}")
            return str(next(SeqIO.parse(fallback_fasta, "fasta")).seq)
        raise RuntimeError(
            f"Could not download {accession}: {e}. "
            "Save the FASTA manually from uniprot.org/uniprotkb/P04626 and pass fallback_fasta."
        )


def clean_sequence(seq: str) -> str:
    """Upper-case and drop non-standard residues (X, B, Z, U, O ...)."""
    return "".join(a for a in seq.upper() if a in AA)


# ----------------------------------------------------------------------
# 2. Features
# ----------------------------------------------------------------------
def aac(seq: str) -> dict:
    """Amino acid composition: 20 features (fraction of each residue)."""
    n = len(seq)
    return {f"AAC_{a}": seq.count(a) / n for a in AA}


def dpc(seq: str) -> dict:
    """Dipeptide composition: 400 features (fraction of each overlapping pair)."""
    n = len(seq) - 1
    counts = dict.fromkeys(DIPEPTIDES, 0)
    for i in range(n):
        counts[seq[i:i + 2]] += 1
    return {f"DPC_{dp}": c / n for dp, c in counts.items()}


# Residue groups for group-composition features
GROUPS = {
    "hydrophobic": "AVLIMFWC",
    "polar_uncharged": "STNQYGPH",
    "positive": "KRH",
    "negative": "DE",
    "small": "AGSTCPDN",
    "tiny": "AGSC",
    "aliphatic": "AVLI",
    "aromatic": "FWY",
}


def physchem(seq: str) -> dict:
    """Global physicochemical descriptors."""
    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    feats = {
        "PC_length": len(seq),
        "PC_log_length": float(np.log10(len(seq))),
        "PC_mol_weight": pa.molecular_weight(),
        "PC_pI": pa.isoelectric_point(),
        "PC_gravy": pa.gravy(),                       # hydrophobicity (Kyte-Doolittle)
        "PC_aromaticity": pa.aromaticity(),
        "PC_instability_index": pa.instability_index(),
        "PC_charge_pH7": pa.charge_at_pH(7.0),
        "PC_frac_helix": helix,
        "PC_frac_turn": turn,
        "PC_frac_sheet": sheet,
        "PC_extinction_reduced": pa.molar_extinction_coefficient()[0],
        "PC_extinction_oxidized": pa.molar_extinction_coefficient()[1],
        "PC_flexibility_mean": float(np.mean(pa.flexibility())),
    }
    for name, residues in GROUPS.items():
        feats[f"PC_frac_{name}"] = sum(seq.count(a) for a in residues) / len(seq)
    feats["PC_net_charge_ratio"] = (
        (seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")) / len(seq)
    )
    return feats


def protein_feature_vector(seq: str) -> pd.Series:
    seq = clean_sequence(seq)
    feats = {**aac(seq), **dpc(seq), **physchem(seq)}
    return pd.Series(feats)


# ----------------------------------------------------------------------
# 3. Repeat for all 1,000 drug-protein pairs
# ----------------------------------------------------------------------
def build_pair_matrix(pairs: pd.DataFrame, vec: pd.Series) -> pd.DataFrame:
    """
    pairs : DataFrame with one row per drug-protein pair (1,000 rows).
    vec   : the single ERBB2 feature vector.
    Returns the pairs table with the ERBB2 features appended to every row.
    """
    prot_block = pd.DataFrame(
        np.tile(vec.values, (len(pairs), 1)),
        columns=vec.index,
        index=pairs.index,
    )
    return pd.concat([pairs.reset_index(drop=True), prot_block.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    # --- 1) sequence ---
    seq = get_uniprot_sequence("P04626", fallback_fasta="P04626.fasta")

    # --- 2) ONE ERBB2 feature vector ---
    vec = protein_feature_vector(seq)
    print(f"Feature vector length: {len(vec)} "
          f"(AAC=20, DPC=400, physchem={len(vec) - 420})")
    vec.to_frame("ERBB2").to_csv("erbb2_feature_vector.csv")

    # --- 3) repeat for 1,000 pairs ---
    if os.path.exists("drug_protein_pairs.csv"):
        pairs = pd.read_csv("drug_protein_pairs.csv")   # e.g. drug_id, smiles, label
    else:  # placeholder so the script runs end to end
        pairs = pd.DataFrame({"drug_id": [f"drug_{i}" for i in range(1000)],
                              "protein_id": "P04626"})

    X = build_pair_matrix(pairs, vec)
    X.to_csv("pair_features_erbb2.csv", index=False)
    print("Pair feature matrix:", X.shape)   # (1000, n_pair_cols + 420 + physchem)
