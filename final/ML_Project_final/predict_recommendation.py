#predict_recommendation.py
import os, joblib, json, pandas as pd
from sklearn.neighbors import NearestNeighbors

#artifacts live right here
HERE          = os.getcwd()
KNN           = joblib.load(os.path.join(HERE, "knn_model.joblib"))
FEATURE_COLS  = json.load(open(os.path.join(HERE, "rec_feature_columns.json")))

def recommend(csv_path:str, m:int=5, k:int=50) -> pd.DataFrame:
    """
    1) load train_recs + predictions.csv → rebuild train Mechanisms & Success
    2) load test_recs + predictions.csv → features + Mechanisms + pred_int/P_*
    3) for each test row, find k neighbors, compute success‐rate per mechanism
    4) return top‐m new recommendations + drop‐prob
    """
    #train side
    train = pd.read_csv("train_recs.csv") 
    train_preds = pd.read_csv("predictions.csv")[[
        "Student_id","pred_int","P_low","P_med","P_high"
    ]]
    train = train.merge(train_preds, on="Student_id", how="left")
    train["Mechanisms"] = train["Stress Coping Mechanisms"].str.split(",")
    # define “success” = ended in Low stress
    train["Success"] = (train["pred_int"] == 0).astype(int)

    #test side 
    test_recs = pd.read_csv("test_recs.csv")
    test_preds= pd.read_csv("predictions.csv")[[
        "Student_id","pred_int","P_low","P_med","P_high"
    ]]
    df = test_recs.merge(test_preds, on="Student_id", how="left")
    df["Mechanisms"] = df["Stress Coping Mechanisms"].str.split(",")

    #recommendation helper
    def _rec(row):
        x = row[FEATURE_COLS].values.reshape(1,-1)
        _, idxs = KNN.kneighbors(x, n_neighbors=k)
        neigh = train.iloc[idxs[0]]
        stats = {}
        for mechs, succ in zip(neigh["Mechanisms"], neigh["Success"]):
            for mech in mechs:
                stats.setdefault(mech, {"used":0,"succ":0})
                stats[mech]["used"] += 1
                stats[mech]["succ"]  += succ
        mech_df = pd.DataFrame([
            {"Mechanism":m, "SuccessRate":v["succ"]/v["used"]}
            for m,v in stats.items()
        ])
        #exclude those they already use
        already = set(row["Mechanisms"])
        mech_df = mech_df[~mech_df["Mechanism"].isin(already)]
        return ",".join(
            mech_df.sort_values("SuccessRate", ascending=False)
                  .head(m)["Mechanism"]
        )

    df["recommendations"] = df.apply(_rec, axis=1)

    #compute drop probability
    def _pdrop(r):
        if r.pred_int == 2:  # high → P_low + P_med
            return r.P_low + r.P_med
        if r.pred_int == 1:  # mdeium → P_low
            return r.P_low
        return 0.0             #low → 0
    df["P_category_drop"] = df.apply(_pdrop, axis=1)

    return df

# when run as script,dump the result
if __name__ == "__main__":
    out = recommend("test_recs.csv")
    out.to_csv("knn_recommendations.csv", index=False)
    print("✅ Wrote knn_recommendations.csv")
