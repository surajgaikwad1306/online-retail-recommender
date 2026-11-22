"""Project 39 - Online Retail Product Recommendation.

Core library: data cleaning, implicit-rating construction, recommenders
(Popularity, Content-Based, Item-Based CF, Matrix Factorisation (SVD), Hybrid)
and top-N ranking metrics.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

NON_PRODUCT_CODES = {"POST", "D", "M", "C2", "DOT", "BANK CHARGES", "CRUK", "PADS",
                     "AMAZONFEE", "S", "B", "m", "gift_0001_10", "gift_0001_20",
                     "gift_0001_30", "gift_0001_40", "gift_0001_50"}


# ----------------------------------------------------------------------------- data
def load_clean(path):
    df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": "float"})
    raw = len(df)
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[~df["InvoiceNo"].str.startswith("C")]           # cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]   # returns / errors / free items
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]       # postage, fees, manual adjustments
    df = df.drop_duplicates()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    df["Description"] = df["Description"].str.strip().str.upper()
    return df, raw


def temporal_split(df, cutoff):
    cutoff = pd.Timestamp(cutoff)
    return df[df["InvoiceDate"] < cutoff], df[df["InvoiceDate"] >= cutoff]


def item_catalogue(df):
    """One description per StockCode (most frequent)."""
    return (df.groupby(["StockCode", "Description"]).size().reset_index(name="n")
              .sort_values("n", ascending=False).drop_duplicates("StockCode")
              .set_index("StockCode")["Description"])


def implicit_ratings(train):
    """Implicit rating r_ui = 1 + log(number of distinct invoices containing the item)
    + 0.5*log(1+total quantity). Retail data has no explicit stars, so repeat-purchase
    frequency and volume are used as preference strength."""
    g = train.groupby(["CustomerID", "StockCode"]).agg(
        n_inv=("InvoiceNo", "nunique"), qty=("Quantity", "sum")).reset_index()
    g["rating"] = 1 + np.log(g["n_inv"]) + 0.5 * np.log1p(g["qty"])
    return g


class Data:
    """Index maps + sparse user-item matrix built from the training period only."""

    def __init__(self, train):
        r = implicit_ratings(train)
        self.users = np.sort(r["CustomerID"].unique())
        self.items = np.sort(r["StockCode"].unique())
        self.uidx = {u: i for i, u in enumerate(self.users)}
        self.iidx = {s: i for i, s in enumerate(self.items)}
        rows = r["CustomerID"].map(self.uidx).values
        cols = r["StockCode"].map(self.iidx).values
        self.R = sparse.csr_matrix((r["rating"].values, (rows, cols)),
                                   shape=(len(self.users), len(self.items)))
        self.B = (self.R > 0).astype(np.float32)
        cat = item_catalogue(train)
        self.desc = cat.reindex(self.items).fillna("").values
        self.price = train.groupby("StockCode")["UnitPrice"].median().reindex(self.items).values


# ----------------------------------------------------------------------------- models
class Popularity:
    name = "Popularity"

    def fit(self, d):
        self.pop = np.asarray(d.B.sum(axis=0)).ravel()
        return self

    def score(self, d, u):
        return np.tile(self.pop, (len(u), 1)).astype(float)


class ContentBased:
    """TF-IDF on product descriptions (word 1-2 grams); user profile = rating-weighted
    mean of the TF-IDF vectors of purchased items; score = cosine(profile, item)."""
    name = "Content-Based"

    def fit(self, d):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.X = normalize(vec.fit_transform(d.desc))
        self.vocab = vec.get_feature_names_out()
        return self

    def score(self, d, u):
        prof = normalize(d.R[u] @ self.X)
        return (prof @ self.X.T).toarray()

    def similar(self, i, k=10):
        s = (self.X[i] @ self.X.T).toarray().ravel()
        s[i] = -1
        return np.argsort(-s)[:k], s


class ItemCF:
    """Item-based collaborative filtering: cosine similarity of item columns of the
    binary user-item matrix, keep top-`nn` neighbours per item, score = R @ S."""
    name = "Item-CF"

    def __init__(self, nn=100):
        self.nn = nn

    def fit(self, d):
        Xn = normalize(d.B.T.tocsr())
        S = (Xn @ Xn.T).toarray()
        np.fill_diagonal(S, 0)
        if self.nn:
            thr = -np.partition(-S, self.nn, axis=1)[:, self.nn][:, None]
            S[S < thr] = 0
        self.S = S
        return self

    def score(self, d, u):
        return np.asarray(d.R[u] @ self.S)


class MatrixFactorisation:
    """Truncated SVD (latent-factor MF) of the implicit-rating matrix."""
    name = "Matrix Factorisation (SVD)"

    def __init__(self, k=50, seed=42):
        self.k, self.seed = k, seed

    def fit(self, d):
        self.svd = TruncatedSVD(n_components=self.k, random_state=self.seed)
        self.U = self.svd.fit_transform(d.R)
        self.V = self.svd.components_
        return self

    def score(self, d, u):
        return self.U[u] @ self.V


def _minmax(S):
    lo, hi = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    return (S - lo) / np.where(hi - lo == 0, 1, hi - lo)


class Hybrid:
    """Weighted hybrid: per-user min-max normalised scores of the component models."""
    name = "Hybrid"

    def __init__(self, models, weights):
        self.models, self.weights = models, weights

    def fit(self, d):
        for m in self.models:
            m.fit(d)
        return self

    def score(self, d, u, parts=None):
        parts = parts or [m.score(d, u) for m in self.models]
        return sum(w * _minmax(p) for w, p in zip(self.weights, parts))


# ----------------------------------------------------------------------------- evaluation
def ground_truth(d, test):
    """Relevant items per user = items bought in test period that are in the training
    catalogue and NOT already bought by that user in training (new-item discovery)."""
    t = test[test["CustomerID"].isin(d.uidx) & test["StockCode"].isin(d.iidx)]
    gt = {}
    for cid, grp in t.groupby("CustomerID"):
        u = d.uidx[cid]
        seen = set(d.B[u].indices)
        rel = {d.iidx[s] for s in grp["StockCode"].unique()} - seen
        if rel:
            gt[u] = rel
    return gt


def topk(d, scores, users, k):
    S = scores.copy()
    for row, u in enumerate(users):
        S[row, d.B[u].indices] = -np.inf     # never re-recommend already-bought items
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def evaluate(d, scores, users, gt, k=10):
    recs = topk(d, scores, users, k)
    disc = 1 / np.log2(np.arange(2, k + 2))
    P, Rc, H, N, per_user = [], [], [], [], []
    for row, u in enumerate(users):
        rel = gt[u]
        hits = np.array([i in rel for i in recs[row]], dtype=float)
        p, r = hits.sum() / k, hits.sum() / len(rel)
        ndcg = (hits * disc).sum() / disc[:min(len(rel), k)].sum()
        P.append(p); Rc.append(r); H.append(float(hits.any())); N.append(ndcg)
        per_user.append((u, p, r, ndcg))
    cov = len(np.unique(recs)) / len(d.items)
    return dict(precision=np.mean(P), recall=np.mean(Rc), hit_rate=np.mean(H),
                ndcg=np.mean(N), coverage=cov), recs, per_user
