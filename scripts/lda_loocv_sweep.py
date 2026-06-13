"""Quick experiment: sweep the PCA-pre-reduction dimensionality for a
PCA->LDA pipeline and report leave-one-out CV accuracy on the labeled routing
set. 1024 features / 16 samples is hopelessly underdetermined for raw LDA, so we
reduce with PCA first (a standard fix) and find the k that generalizes best.

Reads precomputed embeddings from cmd/embeddump. No network."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

emb_path = sys.argv[1] if len(sys.argv) > 1 else "data/routing_embeddings.json"
doc = json.loads(Path(emb_path).read_text())
items = doc["items"]
X = np.array([it["embedding"] for it in items], dtype=float)
y = np.array([1 if (it.get("label") or it.get("path")) == "agentic" else 0 for it in items])
n = len(X)
print(f"{n} samples, {X.shape[1]} features, {int(y.sum())} agentic / {int((1-y).sum())} chat\n")


def loocv(pipeline_fn):
    correct = 0
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        if len(set(y[tr].tolist())) < 2:
            continue
        pred = pipeline_fn(X[tr], y[tr], X[i])
        if pred == y[i]:
            correct += 1
    return correct


def make_pca_lda(k):
    def fn(Xtr, ytr, xq):
        p = PCA(n_components=min(k, Xtr.shape[0] - 1))
        Ztr = p.fit_transform(Xtr)
        lda = LinearDiscriminantAnalysis()
        lda.fit(Ztr, ytr)
        return int(lda.predict(p.transform(xq.reshape(1, -1)))[0])
    return fn


def make_centroid_cosine():
    """Baseline: nearest class mean by cosine in raw embedding space."""
    def fn(Xtr, ytr, xq):
        def unit(v):
            return v / (np.linalg.norm(v) + 1e-12)
        c0 = unit(Xtr[ytr == 0].mean(axis=0))
        c1 = unit(Xtr[ytr == 1].mean(axis=0))
        q = unit(xq)
        return 1 if (q @ c1) > (q @ c0) else 0
    return fn


print("PCA->LDA leave-one-out CV:")
best = (0, None)
for k in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
    c = loocv(make_pca_lda(k))
    print(f"  k={k:2d}: {c}/{n} = {c/n:.1%}")
    if c > best[0]:
        best = (c, k)
print(f"  best: k={best[1]} at {best[0]}/{n} = {best[0]/n:.1%}\n")

c = loocv(make_centroid_cosine())
print(f"nearest-class-mean (cosine, raw embeddings) LOOCV: {c}/{n} = {c/n:.1%}")
