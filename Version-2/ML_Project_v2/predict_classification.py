import os, joblib, json, pandas as pd

# ── load all artifacts from the current working directory ─────────────────
ART = os.getcwd()  

# you only really need the pipeline (which already contains your scaler)
svm_model       = joblib.load(os.path.join(ART, "svm_model.joblib"))
label_map       = json.load(open(os.path.join(ART, "label_map.json")))
feature_columns = json.load(open(os.path.join(ART, "feature_columns.json")))

# flip the JSON’s string‐keys into ints, keep values as labels
inv_map = { int(k):v for k,v in label_map.items() }

def load_and_classify(csv_path: str) -> pd.DataFrame:
    """
    1) read unseen CSV
    2) reconstruct & drop any one-hot dummies
    3) drop unused columns
    4) map Yes/No → 0/1
    5) select exactly feature_columns (still a DataFrame!)
    6) let the pipeline scale & predict_proba
    7) return DataFrame of pred_int, pred_label, P_low/med/high
    """
    df = pd.read_csv(csv_path)

    # 2) if one-hot dummies for target exist, rebuild & drop them
    target = "Stress Level Category"
    dummies = [c for c in df.columns if c.startswith(f"{target}_")]
    if dummies:
        df[target] = (
            df[dummies]
              .idxmax(axis=1)
              .str.replace(f"{target}_","",regex=False)
        )
        df.drop(columns=dummies, inplace=True)

    # 3) drop text/id columns
    for col in ["Stress Coping Mechanisms","Student ID","Unnamed: 0"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # 4) map yes/no → 0/1
    for col in ["Counseling Attendance","Family Mental Health History","Medical Condition"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes":1,"No":0})

    # 5) select exactly the features you trained on (still a DataFrame!)
    X_df = df[feature_columns]

    # 6) let the pipeline do scaling + prediction
    probs = svm_model.predict_proba(X_df)    # no scaler.transform here!
    preds = probs.argmax(axis=1)

    # 7) return results
    return pd.DataFrame({
        "pred_int"  : preds,
        "pred_label": [inv_map[p] for p in preds],
        "P_low"     : probs[:,0],
        "P_med"     : probs[:,1],
        "P_high"    : probs[:,2],
    })
