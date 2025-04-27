#predict_classification.py

import os
import joblib
import json
import pandas as pd
import warnings

# load artifacts at import time
HERE            = os.getcwd()   # should be ML_Project folder
scaler          = joblib.load(os.path.join(HERE, "scaler.joblib"))
imputer         = joblib.load(os.path.join(HERE, "imputer.joblib"))
rf_model        = joblib.load(os.path.join(HERE, "rf_model.joblib"))

_raw_label_map  = json.load(open(os.path.join(HERE, "label_map.json")))
feature_columns = json.load(open(os.path.join(HERE, "feature_columns.json")))

# build inv_map (int→string) reliably
# If the JSON contains string→int, invert it.
# Otherwise if it already contains {"0":"Low",…}just cast keys
if all(isinstance(v, int) for v in _raw_label_map.values()):
    #raw is {'Low':0,'Medium':1,'High':2}
    inv_map = {v: k for k, v in _raw_label_map.items()}
else:
    #raw is {'0':'Low','1':'Medium','2':'High'}
    inv_map = {int(k): v for k, v in _raw_label_map.items()}


def load_and_classify(csv_path: str) -> pd.DataFrame:
    """
    1) Read csv_path
    2) Reconstruct & drop any one-hot dummies of the target
    3) Drop unused text/ID columns
    4) Map Yes/No → 0/1
    5) Select exactly feature_columns → impute → scale → predict_proba
    6) Return DataFrame with pred_int, pred_label, P_low, P_med, P_high
    """
    df = pd.read_csv(csv_path)

    #if one-hot dummies for the target exist, rebuild & drop them
    target = "Stress Level Category"
    dummies = [c for c in df.columns if c.startswith(f"{target}_")]
    if dummies:
        df[target] = (
            df[dummies]
              .idxmax(axis=1)
              .str.replace(f"{target}_", "", regex=False)
        )
        df.drop(columns=dummies, inplace=True)

    #drop raw text/ID columns
    for col in ["Stress Coping Mechanisms", "Student ID", "Unnamed: 0"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    #map Yes/No → 0/1
    for col in ["Counseling Attendance",
                "Family Mental Health History",
                "Medical Condition"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes":1, "No":0})

    #select exactly the features we trained on
    X_df = df[feature_columns]

    #impute via NumPy array to avoid feature-name warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        X_imp = imputer.transform(X_df.values)

    #scale & predict
    Xs    = scaler.transform(X_imp)
    probs = rf_model.predict_proba(Xs)
    preds = probs.argmax(axis=1)

    #build output DataFrame
    return pd.DataFrame({
        "pred_int"  : preds,
        "pred_label": [inv_map[p] for p in preds],
        "P_low"     : probs[:,0],
        "P_med"     : probs[:,1],
        "P_high"    : probs[:,2],
    })
