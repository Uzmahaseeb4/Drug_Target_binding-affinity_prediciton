import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

X = np.load("X_pairs.npy")
y = np.load("y_pIC50.npy")
groups = pd.read_csv("pair_dataset.csv", usecols=["molecule_chembl_id"])["molecule_chembl_id"]

# 80/20 split, no molecule shared between train and test
tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(X, y, groups))

models = {
    "RandomForest": RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                          n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.5,
                            tree_method="hist", random_state=42),
    # SVR needs scaling: standardise only the protein columns (after the 2048 bits)
    "SVR": Pipeline([
        ("scale", ColumnTransformer([("p", StandardScaler(), slice(2048, None))],
                                    remainder="passthrough")),
        ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
    ]),
}

for name, m in models.items():
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    print(f"{name:13s} R2={r2_score(y[te], p):.3f}  "
          f"RMSE={mean_squared_error(y[te], p) ** 0.5:.3f}  "
          f"MAE={mean_absolute_error(y[te], p):.3f}")