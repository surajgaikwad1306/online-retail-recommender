"""Working prototype: personalised product recommendations for a customer, or
'customers also bought / similar products' for a product.

  python recommend.py --customer 17850 --k 10
  python recommend.py --customer 17850 --model item-cf
  python recommend.py --product 85123A --k 8
  python recommend.py --new-basket 22423 85123A 47566     # cold-start customer
"""
import argparse
import sys

import numpy as np
from scipy import sparse

from recsys import (Data, Popularity, ContentBased, ItemCF, MatrixFactorisation, Hybrid,
                    load_clean, _minmax)

WEIGHTS = [0.4, 0.2, 0.3, 0.1]   # cf, mf, cb, pop - tuned on validation period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/retail.csv")
    ap.add_argument("--customer", type=int)
    ap.add_argument("--product")
    ap.add_argument("--new-basket", nargs="+", metavar="STOCKCODE")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="hybrid",
                    choices=["hybrid", "item-cf", "mf", "content", "popularity"])
    a = ap.parse_args()

    df, _ = load_clean(a.data)
    d = Data(df)                                  # production mode: all history
    cf, mf, cb, pop = ItemCF(50), MatrixFactorisation(20), ContentBased(), Popularity()
    hyb = Hybrid([cf, mf, cb, pop], WEIGHTS).fit(d)
    pick = {"item-cf": cf, "mf": mf, "content": cb, "popularity": pop, "hybrid": hyb}[a.model]
    line = "-" * 86

    if a.product:
        if a.product not in d.iidx:
            sys.exit(f"Unknown StockCode {a.product}")
        i = d.iidx[a.product]
        print(f"PRODUCT {a.product}: {d.desc[i]}  (£{d.price[i]:.2f})\n{line}")
        co = cf.S[i].copy()
        print("Customers who bought this also bought (Item-CF cosine):")
        for r, j in enumerate(np.argsort(-co)[:a.k], 1):
            print(f" {r:2d}. {d.items[j]:<8} {d.desc[j][:48]:<48} £{d.price[j]:6.2f}  sim={co[j]:.3f}")
        idx, s = cb.similar(i, a.k)
        print(f"{line}\nSimilar products by description (Content-Based TF-IDF cosine):")
        for r, j in enumerate(idx, 1):
            print(f" {r:2d}. {d.items[j]:<8} {d.desc[j][:48]:<48} £{d.price[j]:6.2f}  sim={s[j]:.3f}")
        return

    if a.new_basket:                              # cold-start: build a pseudo user row
        cols = [d.iidx[s] for s in a.new_basket if s in d.iidx]
        if not cols:
            sys.exit("None of those StockCodes are known")
        row = sparse.csr_matrix((np.ones(len(cols)), ([0] * len(cols), cols)),
                                shape=(1, len(d.items)))
        d.R = sparse.vstack([d.R, row]).tocsr(); d.B = (d.R > 0).astype(np.float32)
        mf.U = np.vstack([mf.U, mf.svd.transform(row)])
        u, header = d.R.shape[0] - 1, "NEW CUSTOMER (cold start, basket only)"
    else:
        if a.customer not in d.uidx:
            sys.exit(f"Customer {a.customer} not found - try --new-basket for cold start")
        u, header = d.uidx[a.customer], f"CUSTOMER {a.customer}"
        country = df.loc[df.CustomerID == a.customer, "Country"].iloc[0]
        header += f"  ({country}, {df[df.CustomerID == a.customer].InvoiceNo.nunique()} orders)"

    hist = d.R[u].toarray().ravel()
    print(f"{header}  model={a.model}\n{line}\nTop items in purchase history:")
    for j in np.argsort(-hist)[:5]:
        if hist[j] > 0:
            print(f"   - {d.items[j]:<8} {d.desc[j][:55]}  (pref={hist[j]:.2f})")

    parts = [m.score(d, [u]) for m in (cf, mf, cb, pop)]
    s = hyb.score(d, [u], parts) if a.model == "hybrid" else pick.score(d, [u])
    s = s.ravel().astype(float); s[d.B[u].indices] = -np.inf
    top = np.argsort(-s)[:a.k]
    norm = [_minmax(p).ravel() for p in parts]
    print(f"{line}\nRecommended for you (top {a.k}):")
    print(f" {'#':>2}  {'Code':<8} {'Product':<44} {'Price':>7}  {'score':>5}  CF / MF / CB / POP")
    for r, j in enumerate(top, 1):
        why = " / ".join(f"{n[j]:.2f}" for n in norm)
        print(f" {r:2d}. {d.items[j]:<8} {d.desc[j][:44]:<44} £{d.price[j]:6.2f}  {s[j]:.3f}  {why}")


if __name__ == "__main__":
    main()
