"""Quantify the embedding-compression tradeoff for the retrieval index.

Compresses the node embeddings (data/node_embeddings.json, 1024-d Jina) to K
dimensions with PCA and measures how well K-dim retrieval preserves the full-dim
nearest-neighbour structure (recall@k). This is the memory<->recall curve: PCA is
a real memory win for the retrieval index, at a measurable recall cost — the
honest version of "compress the embeddings to save memory".

Also evaluates a WHITENED variant (each PCA component / sqrt(eigenvalue)).
Euclidean distance in PCA-whitened space equals Mahalanobis distance in the
original space, so this directly tests the "Mahalanobis / KNN on PCA" idea: does
decorrelating the axes help or hurt neighbour recall?

Ground truth: for each node, its top-k cosine neighbours in the FULL 1024-d
space. recall@k = mean over nodes of |topk_compressed ∩ topk_full| / k.

Pure Python (numpy + scikit-learn). No network, no API key.

Usage: python scripts/compression_curve.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def topk(Xn, k):
    """Indices of the top-k cosine neighbours (rows L2-normalised), excluding self."""
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, kth=k - 1, axis=1)[:, :k]


def recall_at_k(full_idx, comp_idx, k):
    tot = 0.0
    for i in range(full_idx.shape[0]):
        tot += len(set(full_idx[i].tolist()) & set(comp_idx[i].tolist())) / k
    return tot / full_idx.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="data/node_embeddings.json")
    ap.add_argument("--out", default="data/compression_curve.md")
    ap.add_argument("--k", type=int, default=10, help="neighbours for recall@k")
    args = ap.parse_args()

    doc = json.loads(Path(args.embeddings).read_text())
    emb = doc.get("embeddings", doc) if isinstance(doc, dict) else doc
    ids = list(emb.keys())
    X = np.array([emb[i] for i in ids], dtype=float)
    n, d = X.shape
    k = min(args.k, n - 1)

    full_idx = topk(l2norm(X), k)

    rows = []
    kmax = min(n, d)  # PCA components are bounded by min(n_samples, n_features)
    for K in [c for c in [8, 16, 32, 64, 128, 256, 512] if c < kmax]:
        pca = PCA(n_components=K, random_state=0).fit(X)
        Z = pca.transform(X)
        rec_cos = recall_at_k(full_idx, topk(l2norm(Z), k), k)
        Zw = Z / np.sqrt(pca.explained_variance_ + 1e-12)  # whiten = Mahalanobis
        rec_whit = recall_at_k(full_idx, topk(l2norm(Zw), k), k)
        evr = float(pca.explained_variance_ratio_.sum())
        rows.append((K, K * 4, d / K, evr, rec_cos, rec_whit))

    lines = [
        "# Embedding compression curve (PCA)\n",
        f"Corpus: {n} node embeddings, {d}-d (Jina). recall@{k} measured against the "
        f"full-dim cosine neighbours. \"Whitened\" = Euclidean in PCA-whitened space "
        f"(= Mahalanobis in the original space).\n",
        f"| dims (K) | bytes/node | compression | variance kept | recall@{k} (PCA cosine) | recall@{k} (PCA whitened) |",
        "|---|---|---|---|---|---|",
        f"| {d} (full) | {d * 4} | 1.0x | 100% | 1.000 | 1.000 |",
    ]
    for K, b, ratio, evr, rc, rw in rows:
        lines.append(f"| {K} | {b} | {ratio:.0f}x | {evr * 100:.0f}% | {rc:.3f} | {rw:.3f} |")
    md = "\n".join(lines) + "\n"
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
