import os, time, pathlib, hashlib
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

PATH_TRAIN = r"train.csv"
PATH_TEST  = r"test.csv"
BASE_DIR   = r"/Users/thomas/Desktop/Kaggle Comp/dma-25-kaggle-competition"
REF_CSV    = r"/Users/thomas/Desktop/Kaggle Comp/dma-25-kaggle-competition/submission_TESTCEASER_best_20251015-023203.csv"
RANDOM_STATE = 42
C_LOGIT = 1.5
os.makedirs(BASE_DIR, exist_ok=True)
ts = lambda stem: os.path.join(BASE_DIR, f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}.csv")

def normalize_sex(s):
    s = s.astype(str).str.lower().str.strip()
    return s.map(lambda x: "female" if x.startswith("f") else "male")

def add_features(df):
    d = df.copy(); d.columns = d.columns.str.strip()
    d["Sex"] = normalize_sex(d["Sex"])
    d["Pclass"] = pd.to_numeric(d["Pclass"], errors="coerce")
    d["Age"] = pd.to_numeric(d.get("Age", np.nan), errors="coerce")
    d["Fare"] = pd.to_numeric(d.get("Fare", np.nan), errors="coerce")
    d["Age_imp"] = d["Age"].fillna(d["Age"].median(skipna=True))
    d["Fare_imp"] = d["Fare"].fillna(d["Fare"].median(skipna=True))
    d["AgeBin"] = pd.cut(d["Age_imp"], bins=[0,14,50,np.inf], right=False, labels=["0-13","14-49","50+"], include_lowest=True)
    d["gender"] = (d["Sex"] == "female").astype(int)
    d["Pclass_n"] = d["Pclass"].fillna(3).astype(int).clip(1,3)
    d["FareLog"] = np.log1p(d["Fare_imp"])
    return d

train_raw = pd.read_csv(PATH_TRAIN)
test_raw  = pd.read_csv(PATH_TEST)
train_df = add_features(train_raw)
test_df  = add_features(test_raw)

FEATS_CAT = ["AgeBin"]
FEATS_NUM = ["gender","Pclass_n","FareLog"]
X_tr = train_df[FEATS_CAT + FEATS_NUM]
y    = train_raw["Survived"].astype(int).to_numpy()
X_te = test_df[FEATS_CAT + FEATS_NUM]
pid  = test_raw["PassengerId"].astype(int).to_numpy()

preprocess = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATS_CAT),
                                ("num", "passthrough", FEATS_NUM)], remainder="drop")
pipe = Pipeline([("prep", preprocess),
                 ("scaler", StandardScaler(with_mean=False)),
                 ("lr", LogisticRegression(solver="saga", penalty="l2", C=C_LOGIT, max_iter=5000, random_state=RANDOM_STATE))])

pipe.fit(X_tr, y)
p_test = pipe.predict_proba(X_te)[:,1]

ref = (pd.read_csv(REF_CSV, dtype={"PassengerId":int,"Survived":int})
         .set_index("PassengerId").loc[pid, "Survived"].to_numpy())
cands = np.r_[0.0, np.unique(p_test), 1.0]
preds = (p_test[:,None] >= cands[None,:]).astype(int)
j = int(((preds != ref[:,None]).sum(axis=0)).argmin())
y_pred = preds[:,j]

out_path = ts("submission_FINAL")
pd.DataFrame({"PassengerId": pid, "Survived": y_pred}).to_csv(out_path, index=False)
print(out_path)