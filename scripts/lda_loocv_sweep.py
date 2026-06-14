"""Sweep the PCA-pre-reduction dimensionality for a PCA->LDA routing classifier
and report leave-one-out CV accuracy on the labeled routing set. 1024 features
with a few-dozen samples is underdetermined for raw LDA, so we reduce with PCA
first (a standard fix) and find the k that generalizes best.

LOOCV is leakage-free: for each held-out point, PCA *and* LDA are refit on the
remaining samples only, and the held-out point is just transformed/predicted.

Rigor add-ons:
  - Wilson 95% CI on each accuracy (small N => the point estimate alone misleads).
  - A label-permutation test on the best k: shuffling labels should collapse
    LOOCV to ~chance; if it doesn't, the result is an overfitting artifact.

Reads precomputed embeddings (from cmd/embeddump). No network."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

emb_path = sys.argv[1] if len(sys.argv) > 1 else "data/routing_embeddings.json"
doc = json.loads(Path(emb_path).read_text())
items = doc["items"]
X = np.array([it["embedding"] for it in items], dtype=float)
y = np.array([1 if (it.get("label") or it.get("path")) == "agentic" else 0 for it in items])
n = len(X)
print(f"{n} samples, {X.shape[1]} features, {int(y.sum())} agentic / {int((1 - y).sum())} chat\n")


def loocv(pipeline_fn, labels=None):
    """Leave-one-out accuracy (as a count) for a pipeline that refits per fold."""
    if labels is None:
        labels = y
    correct = 0
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        if len(set(labels[tr].tolist())) < 2:
            continue  # a degenerate single-class training fold
        pred = pipeline_fn(X[tr], labels[tr], X[i])
        if pred == labels[i]:
            correct += 1
    return correct


def wilson95(c, n):
    """Wilson score 95% CI for a proportion — honest at small N, unlike the
    normal approximation which can run past [0,1]."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = c / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


def make_pca_lda(k):
    def fn(Xtr, ytr, xq):
        p = PCA(n_components=min(k, Xtr.shape[0] - 1))
        Ztr = p.fit_transform(Xtr)
        lda = LinearDiscriminantAnalysis()
        lda.fit(Ztr, ytr)
        return int(lda.predict(p.transform(xq.reshape(1, -1)))[0])
    return fn


def make_shrinkage_lda():
    def fn(Xtr, ytr, xq):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(Xtr, ytr)
        return int(lda.predict(xq.reshape(1, -1))[0])
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


def permutation_test(pipeline_fn, reps=200, seed=0):
    """Shuffle labels and rerun LOOCV; real signal should sit well above the
    shuffled-label null. Returns (real_acc, null_mean, null_max, p_value)."""
    rng = np.random.default_rng(seed)
    real = loocv(pipeline_fn) / n
    null = np.empty(reps)
    for r in range(reps):
        yp = y.copy()
        rng.shuffle(yp)
        null[r] = loocv(pipeline_fn, yp) / n
    pval = (np.sum(null >= real) + 1) / (reps + 1)
    return real, float(null.mean()), float(null.max()), float(pval)


print("PCA->LDA leave-one-out CV:")
best = (0, None)
for k in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
    c = loocv(make_pca_lda(k))
    lo, hi = wilson95(c, n)
    print(f"  k={k:2d}: {c}/{n} = {c / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})")
    if c > best[0]:
        best = (c, k)
lo, hi = wilson95(best[0], n)
print(f"  best: k={best[1]} at {best[0]}/{n} = {best[0] / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})\n")

c = loocv(make_centroid_cosine())
lo, hi = wilson95(c, n)
print(f"nearest-class-mean (cosine, raw embeddings) LOOCV: {c}/{n} = {c / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})\n")

# --- regularized + bias-free estimates (sklearn handles refit-per-fold) ---
shrink = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
sc = int(cross_val_score(shrink, X, y, cv=LeaveOneOut()).sum())
lo, hi = wilson95(sc, n)
print(f"shrinkage-LDA (raw 1024-d, no PCA)      LOOCV: {sc}/{n} = {sc / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})")

# Nested CV: inner 5-fold picks k, outer LOO scores — removes the optimistic bias
# of choosing 'best k' on the same LOO we report above.
pipe = Pipeline([("pca", PCA()), ("lda", LinearDiscriminantAnalysis())])
grid = GridSearchCV(pipe, {"pca__n_components": [1, 2, 3, 4, 5, 6, 8, 10, 12]},
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0))
nc = int(cross_val_score(grid, X, y, cv=LeaveOneOut()).sum())
lo, hi = wilson95(nc, n)
print(f"PCA->LDA NESTED CV (unbiased k-selection) LOOCV: {nc}/{n} = {nc / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})\n")

def report_permutation(name, pipeline_fn):
    real, null_mean, null_max, pval = permutation_test(pipeline_fn)
    print(f"Label-permutation test — {name} (200 shuffles; chance ~50%):")
    print(f"  real LOOCV:           {real:.1%}")
    print(f"  shuffled-label LOOCV: mean {null_mean:.1%}, max {null_max:.1%}")
    verdict = "real signal (above the shuffled null)" if pval < 0.05 else "NOT distinguishable from chance — likely artifact"
    print(f"  permutation p-value:  {pval:.3f}  -> {verdict}\n")


report_permutation(f"PCA->LDA best k={best[1]}", make_pca_lda(best[1]))
report_permutation("shrinkage-LDA (raw 1024-d)", make_shrinkage_lda())
