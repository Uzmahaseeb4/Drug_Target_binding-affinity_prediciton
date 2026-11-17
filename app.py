"""
Streamlit app: pick a drug + a target -> predicted binding affinity (pIC50 / IC50).

Run (in the MINI_PROJECT folder):
    python -m streamlit run app.py

Needs in the same folder: predict_affinity.py, erbb2_features.py, X_pairs.npy,
y_pIC50.npy, pair_dataset.csv, her2_/cdk4_ic50_compounds_unique.csv (for drug names).
Install: pip install streamlit rdkit scikit-learn xgboost joblib pandas numpy
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

import predict_affinity as pa          # reuses the trained models + feature code

st.set_page_config(page_title="Drug-Target Affinity Predictor", page_icon="💊", layout="wide")

TARGETS = {                      # label shown in the dropdown -> UniProt accession
    "HER2 / ERBB2 (P04626)": "P04626",
    "CDK4 (P11802)": "P11802",
}
ACC_TO_LABEL = {v: k for k, v in TARGETS.items()}


# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading / training models (first run only)...")
def load_models():
    return pa.get_models()


@st.cache_data
def load_drugs():
    """One row per drug with its name, SMILES, and the target(s) it was measured against."""
    pairs = pd.read_csv("pair_dataset.csv",
                        usecols=["molecule_chembl_id", "canonical_smiles", "protein_id", "pIC50"])
    names = []
    for f in ["her2_ic50_compounds_unique.csv", "cdk4_ic50_compounds_unique.csv"]:
        if os.path.exists(f):
            names.append(pd.read_csv(f, usecols=["molecule_chembl_id", "molecule_pref_name"]))
    name_map = {}
    if names:
        n = pd.concat(names).dropna(subset=["molecule_pref_name"])
        name_map = n.drop_duplicates("molecule_chembl_id").set_index("molecule_chembl_id")["molecule_pref_name"].to_dict()

    drugs = (pairs.groupby("molecule_chembl_id")
             .agg(smiles=("canonical_smiles", "first"),
                  targets=("protein_id", lambda s: list(dict.fromkeys(s)))).reset_index())
    drugs["name"] = drugs["molecule_chembl_id"].map(name_map)
    drugs["label"] = np.where(drugs["name"].notna(),
                              drugs["name"].astype(str) + "  (" + drugs["molecule_chembl_id"] + ")",
                              drugs["molecule_chembl_id"])
    return drugs.sort_values("label").reset_index(drop=True), pairs


def draw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(380, 280)) if mol else None


def show_result(res, experimental=None):
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted pIC50", res["pIC50_pred"], help="pIC50 = 9 - log10(IC50 in nM). Higher = stronger.")
    c2.metric("Predicted IC50", f"{res['IC50_nM_pred']:,.1f} nM")
    c3.metric("Affinity class", res["affinity_class"].split(" (")[0].capitalize())

    st.progress(float(np.clip((res["pIC50_pred"] - 4) / 7, 0, 1)),
                text="Potency scale: 4 (very weak) → 11 (extremely potent)")
    st.caption(f"Approximate range: pIC50 {res['approx_range_pIC50']}  (typical test error ±0.65)")

    cols = st.columns(3)
    cols[0].metric("Random Forest", res["RF"])
    cols[1].metric("XGBoost", res["XGB"])
    cols[2].metric("SVR", res["SVR"])

    if experimental is not None:
        st.info(f"Experimental pIC50 in your dataset: **{experimental:.2f}** "
                f"({10 ** (9 - experimental):,.2f} nM). This drug was part of the "
                f"training data, so the prediction is not an independent test.")

    sim = res["max_tanimoto_to_train"]
    if sim is not None:
        (st.success if sim >= 0.5 else st.warning)(
            f"Similarity to training drugs: {sim:.2f} - {res['confidence_note']}")


# ---------------------------------------------------------------- UI
st.title("💊 Drug–Target Binding Affinity Predictor")
st.caption("Random Forest + XGBoost + SVR ensemble · Morgan fingerprint + protein AAC/DPC/physicochemical features")

models = load_models()
drugs, pairs = load_drugs()

tab1, tab2 = st.tabs(["Pick a drug from the dataset", "Enter a new SMILES"])

with tab1:
    left, right = st.columns([1, 1])
    with left:
        label = st.selectbox(f"Drug ({len(drugs):,} available, type to search)", drugs["label"], index=0)
        row = drugs[drugs["label"] == label].iloc[0]

        # designated target(s) of this drug = the target(s) it was measured against
        designated = [ACC_TO_LABEL[a] for a in row["targets"] if a in ACC_TO_LABEL]
        st.caption("Designated target in the dataset: **" + ", ".join(designated) + "**")

        # the full list is offered, with the designated target first
        options = designated + [t for t in TARGETS if t not in designated]
        target_label = st.selectbox("Target protein", options, key=f"target_{row['molecule_chembl_id']}")
        acc = TARGETS[target_label]
        if acc not in row["targets"]:
            st.warning("This is not the target this drug was measured against - "
                       "the model is extrapolating.")
        st.code(row["smiles"], language=None)

    with right:
        img = draw(row["smiles"])
        if img is not None:
            st.image(img, caption=row["label"])

    if st.button("Calculate binding affinity", type="primary", key="btn_pick"):
        res = pa.predict(models, row["smiles"], acc)
        exp = pairs[(pairs["molecule_chembl_id"] == row["molecule_chembl_id"])
                    & (pairs["protein_id"] == acc)]["pIC50"]
        st.subheader(f"{row['label']}  →  {target_label}")
        show_result(res, experimental=float(exp.iloc[0]) if len(exp) else None)

with tab2:
    smi = st.text_input("SMILES", placeholder="e.g. CCOc1cc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1")
    tgt2 = st.selectbox("Target protein", list(TARGETS), key="target_custom")
    if st.button("Calculate binding affinity", type="primary", key="btn_custom"):
        if not smi.strip():
            st.error("Enter a SMILES string.")
        elif Chem.MolFromSmiles(smi.strip()) is None:
            st.error("RDKit could not parse that SMILES.")
        else:
            c1, c2 = st.columns([1, 2])
            c1.image(draw(smi.strip()))
            with c2:
                show_result(pa.predict(models, smi.strip(), TARGETS[tgt2]))

st.divider()
st.caption("pIC50 is a potency measure used as a proxy for binding affinity. The model was trained on "
           "HER2 and CDK4 compounds only; predictions for unrelated chemistry are unreliable.")
