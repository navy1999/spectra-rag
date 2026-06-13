"""Broader model search for the routing classifier, all by leave-one-out CV on
the labeled set. Compares, on the SAME embeddings:
  - PCA->LDA (various k)
  - nearest class mean (cosine)
  - L2-regularized logistic regression (various C), on raw and L2-normalized X
  - LDA on L2-normalized X

Goal: find the most accurate LINEAR classifier whose weight vector we can fold
into the existing (components, mean) router schema with zero Go changes. Reads
precomputed embeddings from cmd/embeddump. No network."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

emb_path = sys.argv[1] if len(sys.argv) > 1 else "data/routing_embeddings.json"
doc = json.loads(Path(emb_path).read_text())
items = doc["items"]
X = np.array([it["embedding"] for it in items], dtype=float)
y = np.array([1 if (it.get("label") or it.get("path")) == "agentic" else 0 for it in items])
n = len(X)


def l2norm(M):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12)


Xn = l2norm(X)
print(f"{n} samples, {X.shape[1]} features, {int(y.sum())} agentic / {int((1-y).sum())} chat\n")


def loocv(fit_predict, data):
    correct = 0
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        if len(set(y[tr].tolist())) < 2:
            continue
        if fit_predict(data[tr], y[tr], data[i]) == y[i]:
            correct += 1
    return correct


def pca_lda(k):
    def fn(Xtr, ytr, xq):
        p = PCA(n_components=min(k, Xtr.shape[0] - 1))
        Ztr = p.fit_transform(Xtr)
        m = LinearDiscriminantAnalysis().fit(Ztr, ytr)
        return int(m.predict(p.transform(xq.reshape(1, -1)))[0])
    return fn


def logreg(C):
    def fn(Xtr, ytr, xq):
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
        return int(m.predict(xq.reshape(1, -1))[0])
    return fn


def centroid_cos(Xtr, ytr, xq):
    c0 = l2norm(Xtr[ytr == 0].mean(axis=0))
    c1 = l2norm(Xtr[ytr == 1].mean(axis=0))
    q = xq / (np.linalg.norm(xq) + 1e-12)
    return 1 if (q @ c1) > (q @ c0) else 0


print("== raw embeddings ==")
for k in [2, 4, 8, 16]:
    c = loocv(pca_lda(k), X)
    print(f"  PCA{k}->LDA     {c}/{n} = {c/n:.1%}")
for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
    c = loocv(logreg(C), X)
    print(f"  logreg C={C:<5} {c}/{n} = {c/n:.1%}")
c = loocv(centroid_cos, X)
print(f"  centroid-cos   {c}/{n} = {c/n:.1%}")

print("\n== L2-normalized embeddings ==")
for k in [2, 4, 8, 16]:
    c = loocv(pca_lda(k), Xn)
    print(f"  PCA{k}->LDA     {c}/{n} = {c/n:.1%}")
for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
    c = loocv(logreg(C), Xn)
    print(f"  logreg C={C:<5} {c}/{n} = {c/n:.1%}")
c = loocv(centroid_cos, Xn)
print(f"  centroid-cos   {c}/{n} = {c/n:.1%}")
