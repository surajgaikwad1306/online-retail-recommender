"""Runs the full Project 39 experiment: EDA, validation tuning, test evaluation,
error analysis and figures. Usage: python run_experiment.py data/retail.csv"""
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys import (Data, Popularity, ContentBased, ItemCF, MatrixFactorisation, Hybrid,
                    load_clean, temporal_split, ground_truth, evaluate)

np.random.seed(42)
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/retail.csv"
OUT = Path("results"); OUT.mkdir(exist_ok=True)
K = 10
VAL_CUT, TEST_CUT = "2011-08-01", "2011-09-01"
log = {}

# ---------------------------------------------------------------- 1. data
df, raw = load_clean(DATA)
log["data"] = dict(raw_rows=raw, clean_rows=len(df), customers=df.CustomerID.nunique(),
                   products=df.StockCode.nunique(), invoices=df.InvoiceNo.nunique(),
                   countries=df.Country.nunique(),
                   date_min=str(df.InvoiceDate.min()), date_max=str(df.InvoiceDate.max()))
ui = df.groupby(["CustomerID", "StockCode"]).size()
log["data"]["user_item_pairs"] = len(ui)
log["data"]["sparsity"] = 1 - len(ui) / (log["data"]["customers"] * log["data"]["products"])
print(json.dumps(log["data"], indent=2))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df.set_index("InvoiceDate").resample("MS")["InvoiceNo"].nunique().plot(ax=ax[0], marker="o")
ax[0].axvline(pd.Timestamp(VAL_CUT), color="orange", ls="--", label="validation cut")
ax[0].axvline(pd.Timestamp(TEST_CUT), color="red", ls="--", label="test cut")
ax[0].set_title("Invoices per month"); ax[0].legend()
items_per_user = df.groupby("CustomerID").StockCode.nunique()
ax[1].hist(np.log10(items_per_user), bins=40, color="#4c72b0")
ax[1].set_title("Distinct products per customer (log10)")
pop = df.groupby("StockCode").CustomerID.nunique().sort_values(ascending=False).values
ax[2].plot(np.cumsum(pop) / pop.sum()); ax[2].set_title("Long tail: cumulative share of purchases")
ax[2].set_xlabel("products ranked by popularity")
plt.tight_layout(); plt.savefig(OUT / "fig1_eda.png", dpi=110); plt.close()


def build_models(k_svd=50, nn=100):
    return dict(pop=Popularity(), cb=ContentBased(), cf=ItemCF(nn), mf=MatrixFactorisation(k_svd))


def run_all(train, test, weights=None, k_svd=50, nn=100):
    d = Data(train)
    gt = ground_truth(d, test)
    users = np.array(sorted(gt))
    models = build_models(k_svd, nn)
    t = {}
    parts = {}
    for key, m in models.items():
        s = time.time(); m.fit(d); parts[key] = m.score(d, users); t[key] = time.time() - s
    return d, gt, users, models, parts, t


# ---------------------------------------------------------------- 2. validation tuning
tr_val, rest = temporal_split(df, VAL_CUT)
val, _ = temporal_split(rest, TEST_CUT)
print("\nTuning on validation period", VAL_CUT, "->", TEST_CUT)
tuning = {"svd_k": {}, "cf_nn": {}}
for k_svd in [20, 50, 100]:
    d, gt, users, _, parts, _ = run_all(tr_val, val, k_svd=k_svd)
    tuning["svd_k"][k_svd] = evaluate(d, parts["mf"], users, gt, K)[0]["ndcg"]
best_k = max(tuning["svd_k"], key=tuning["svd_k"].get)
for nn in [50, 100, 300]:
    d, gt, users, _, parts, _ = run_all(tr_val, val, k_svd=best_k, nn=nn)
    tuning["cf_nn"][nn] = evaluate(d, parts["cf"], users, gt, K)[0]["ndcg"]
best_nn = max(tuning["cf_nn"], key=tuning["cf_nn"].get)
d, gt, users, models, parts, _ = run_all(tr_val, val, k_svd=best_k, nn=best_nn)
grid = []
steps = np.arange(0, 1.01, 0.1)
for w_cf, w_mf, w_cb in itertools.product(steps, steps, steps):
    w_pop = 1 - w_cf - w_mf - w_cb
    if w_pop < -1e-9:
        continue
    w = [w_cf, w_mf, w_cb, max(w_pop, 0)]
    h = Hybrid([models["cf"], models["mf"], models["cb"], models["pop"]], w)
    s = h.score(d, users, [parts["cf"], parts["mf"], parts["cb"], parts["pop"]])
    grid.append((evaluate(d, s, users, gt, K)[0]["ndcg"], w))
grid.sort(key=lambda x: -x[0])
best_w = [round(x, 2) for x in grid[0][1]]
tuning.update(best_svd_k=best_k, best_cf_nn=best_nn, hybrid_weights_cf_mf_cb_pop=best_w,
              hybrid_val_ndcg=grid[0][0], top5_weight_sets=[(round(a, 4), [round(x, 2) for x in w]) for a, w in grid[:5]])
print(json.dumps(tuning, indent=2, default=str))
log["tuning"] = tuning

# ---------------------------------------------------------------- 3. test evaluation
train, test = temporal_split(df, TEST_CUT)
d, gt, users, models, parts, times = run_all(train, test, k_svd=best_k, nn=best_nn)
hyb = Hybrid([models["cf"], models["mf"], models["cb"], models["pop"]], best_w)
s = time.time()
parts["hyb"] = hyb.score(d, users, [parts["cf"], parts["mf"], parts["cb"], parts["pop"]])
times["hyb"] = sum(times[k] for k in ["cf", "mf", "cb", "pop"]) + time.time() - s
rng = np.random.default_rng(42)
parts["rand"] = rng.random((len(users), len(d.items))); times["rand"] = 0.0
names = {"rand": "Random", "pop": "Popularity", "cb": "Content-Based", "cf": "Item-CF",
         "mf": f"MF (SVD, k={best_k})", "hyb": "Hybrid"}
rows, per_user, recs_all = [], {}, {}
for key in ["rand", "pop", "cb", "cf", "mf", "hyb"]:
    m, recs, pu = evaluate(d, parts[key], users, gt, K)
    m["fit_score_time_s"] = times[key]
    rows.append({"model": names[key], **m}); per_user[key] = pu; recs_all[key] = recs
res = pd.DataFrame(rows).set_index("model")
log["test"] = dict(train_rows=len(train), test_rows=len(test), train_users=len(d.users),
                   train_items=len(d.items), eval_users=len(users),
                   avg_relevant_per_user=float(np.mean([len(v) for v in gt.values()])),
                   results=res.round(4).to_dict(orient="index"))
print("\nTEST RESULTS (top-%d, new-item discovery)" % K)
print(res.round(4).to_string())
res.round(4).to_csv(OUT / "test_results.csv")

# bootstrap 95% CI for NDCG of best two models + paired difference
def boot(a, n=2000):
    a = np.asarray(a); idx = rng.integers(0, len(a), (n, len(a)))
    return np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
nd = {k: np.array([x[3] for x in per_user[k]]) for k in per_user}
ci = {names[k]: [round(float(x), 4) for x in boot(nd[k])] for k in nd}
diff = nd["hyb"] - nd["cf"]
log["test"]["ndcg_95ci"] = ci
log["test"]["hybrid_minus_itemcf_ndcg_95ci"] = [round(float(x), 4) for x in boot(diff)]
print("NDCG 95% CI:", ci)
print("Hybrid - ItemCF NDCG diff CI:", log["test"]["hybrid_minus_itemcf_ndcg_95ci"])

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
res[["precision", "recall", "hit_rate", "ndcg"]].plot.bar(ax=ax[0], rot=20)
ax[0].set_title(f"Test-set ranking metrics @ {K}"); ax[0].grid(axis="y", alpha=.3)
res["coverage"].plot.bar(ax=ax[1], rot=20, color="#55a868")
ax[1].set_title("Catalogue coverage of top-10 lists"); ax[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig2_model_comparison.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 4. error analysis
n_train_items = np.asarray(d.B.sum(axis=1)).ravel()
bins = [0, 10, 30, 100, 10_000]
labels = ["1-10", "11-30", "31-100", ">100"]
ea = []
for key in ["pop", "cb", "cf", "mf", "hyb"]:
    pu = pd.DataFrame(per_user[key], columns=["u", "p", "r", "ndcg"])
    pu["bucket"] = pd.cut(n_train_items[pu.u], bins, labels=labels)
    g = pu.groupby("bucket", observed=True)["ndcg"].mean()
    ea.append(g.rename(names[key]))
ea = pd.concat(ea, axis=1)
counts = pd.cut(n_train_items[users], bins, labels=labels).value_counts().reindex(labels)
ea["n_users"] = counts.values
log["error_by_history"] = ea.round(4).to_dict(orient="index")
print("\nNDCG by customer history size:\n", ea.round(4).to_string())
ax = ea.drop(columns="n_users").plot.bar(figsize=(10, 4.5), rot=0)
ax.set_title("NDCG@10 by number of distinct products bought in training (cold-start analysis)")
ax.set_xlabel("customer history (distinct products)"); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig3_error_by_history.png", dpi=110); plt.close()

# popularity bias: mean popularity rank of recommended items
pop_rank = np.argsort(np.argsort(-models["pop"].pop))
log["popularity_bias_mean_rank_of_recs"] = {names[k]: float(pop_rank[recs_all[k]].mean())
                                            for k in recs_all}

# weight sensitivity figure
ws = pd.DataFrame([(a, *w) for a, w in grid], columns=["ndcg", "cf", "mf", "cb", "pop"])
fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
for a, c in zip(ax, ["cf", "mf", "cb", "pop"]):
    ws.groupby(ws[c].round(1))["ndcg"].max().plot(ax=a, marker="o")
    a.set_title(f"best val NDCG vs weight_{c}"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig4_hybrid_weight_sensitivity.png", dpi=110); plt.close()

Path(OUT / "metrics.json").write_text(json.dumps(log, indent=2, default=str))
print("\nSaved results/ (metrics.json, test_results.csv, figures)")
