import pandas as pd
from sklearn.model_selection import train_test_split

RAW = "data/H2.csv"
TRAIN = "src/data/classificationdata/H2_classification_train.csv"
TEST  = "src/data/classificationdata/H2_classification_test.csv"

df = pd.read_csv(RAW)
X = df.drop(columns=["IsCanceled", "ReservationStatus", "ReservationStatusDate"])
y = df["IsCanceled"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def dedup(X_, y_):
    t = X_.assign(IsCanceled=y_).drop_duplicates(keep="first")
    return t.drop(columns="IsCanceled"), t["IsCanceled"]

def filt(X_, y_):
    X_ = X_.copy()
    X_["Children"] = X_["Children"].fillna(0)
    guests = X_["Adults"] + X_["Children"] + X_["Babies"]
    keep = (X_["Adults"] > 0) & (guests > 0)
    X_, y_ = X_[keep], y_[keep]
    keep = X_["Country"].notna()
    X_, y_ = X_[keep], y_[keep]
    keep = X_["MarketSegment"].str.strip() != "Undefined"
    return X_[keep], y_[keep]

X_tr, y_tr = filt(*dedup(X_tr, y_tr))
X_te, y_te = filt(*dedup(X_te, y_te))

def clean_agent(s):
    return s.astype(str).str.strip().replace({"NULL": "No_Agent", "nan": "No_Agent"})

for path, X_, y_ in [(TRAIN, X_tr, y_tr), (TEST, X_te, y_te)]:
    old = pd.read_csv(path)
    old.columns = old.columns.str.strip()
    # safety: the rebuilt rows must match the file row by row
    assert len(old) == len(X_), "row count differs"
    assert (old["LeadTime"].values == X_["LeadTime"].values).all()
    assert (old["IsCanceled"].values == y_.values).all()
    old["Agent"] = clean_agent(X_["Agent"]).values
    old.to_csv(path.replace(".csv", "_with_agent.csv"), index=False)