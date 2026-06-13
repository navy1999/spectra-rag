"""
Fits the PCA projection the router uses, on the SAME embeddings the backend uses
at runtime (Jina by default). Dimension consistency is essential: if this fits on
a different embedder than the server calls, the server's projection silently
falls back to a non-semantic dev sketch.

Reads the corpus (data/papers.json if present, else node names from
data/graph.json), embeds it via the embeddings provider, fits 2-component PCA,
and saves:
  data/pca_model.json      — components (2 x n_features) + mean (n_features)
  data/pca_centroids.json  — regime cluster centers in the 2D projected space
  data/pca_labels.json     — per-item 2D coords (for inspection)

Requires the embeddings key in the environment, matching the backend:
  EMBEDDINGS_API_KEY   (required)
  EMBEDDINGS_BASE_URL  (default https://api.jina.ai/v1)
  EMBEDDINGS_MODEL     (default jina-embeddings-v3)
"""
import argparse
import json
import os
from pathlib import Path

import requests

BASE_URL = os.environ.get("EMBEDDINGS_BASE_URL", "https://api.jina.ai/v1")
MODEL = os.environ.get("EMBEDDINGS_MODEL", "jina-embeddings-v3")
API_KEY = os.environ.get("EMBEDDINGS_API_KEY", "")


def embed(texts, batch_size=32):
    """Embed texts via the configured provider (OpenAI-compatible /embeddings)."""
    if not API_KEY:
        raise SystemExit(
            "EMBEDDINGS_API_KEY is not set. Set it to the same key the backend uses, e.g.\n"
            "  PowerShell:  $env:EMBEDDINGS_API_KEY=\"jina_...\"; python fit_pca.py\n"
            "  bash:        EMBEDDINGS_API_KEY=jina_... python fit_pca.py"
        )
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "input": batch},
            timeout=60,
        )
        resp.raise_for_status()
        out.extend(item["embedding"] for item in resp.json()["data"])
    return out


def load_corpus(papers_path, graph_path):
    """Prefer papers.json (title+abstract); fall back to graph.json node names."""
    if Path(papers_path).exists():
        papers = json.loads(Path(papers_path).read_text())
        if papers:
            items = [
                {"id": p.get("id"), "text": (p.get("title", "") + " ") * 3 + p.get("abstract", "")}
                for p in papers
            ]
            return items, "papers"
    if Path(graph_path).exists():
        graph = json.loads(Path(graph_path).read_text())
        items = [{"id": n.get("id"), "text": n.get("name", "")} for n in graph.get("nodes", [])]
        return items, "graph"
    return [], "none"


def fit_pca(papers_path="data/papers.json", graph_path="data/graph.json", output_dir="data"):
    items, source = load_corpus(papers_path, graph_path)
    if not items:
        print("No corpus found; writing default centroids only.")
        Path(output_dir).mkdir(exist_ok=True)
        Path(f"{output_dir}/pca_centroids.json").write_text(
            json.dumps({"logic": [0.42, -0.18], "creative": [-0.31, 0.29]}, indent=2)
        )
        return

    from sklearn.decomposition import PCA  # imported here so the no-corpus path needs no deps

    print(f"Embedding {len(items)} items from {source} via {MODEL} @ {BASE_URL} ...")
    embeddings = embed([it["text"] for it in items])
    dim = len(embeddings[0])
    print(f"Got {len(embeddings)} embeddings of dimension {dim}.")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    import numpy as np
    from sklearn.cluster import KMeans

    # Centroids FROM THE CORPUS: cluster the embeddings (k=2) and project the
    # cluster centers into PCA space, so centroids sit where the data actually
    # lives. (Earlier semantic-anchor centroids projected to a dead zone near the
    # origin, away from the corpus, which — with mis-scaled thresholds — collapsed
    # the router to always-chat.) Each cluster is then labeled by whichever anchor
    # description it is closest to, so "logic"/"creative" stay interpretable and
    # match the backend's regimeBaseTemp keys (router/pca_router.go). Honest note:
    # on a homogeneous corpus these clusters are topic sub-splits, not a true
    # logic-vs-creative axis.
    anchors = {
        "logic": "A precise, factual, analytical question seeking a correct, deterministic, lookup-style answer grounded in specific facts.",
        "creative": "An open-ended, generative, exploratory or imaginative request inviting free-form, speculative, or creative writing.",
    }
    names = list(anchors.keys())
    emb_arr = np.array(embeddings, dtype=float)
    k = min(len(names), len(emb_arr))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb_arr)
    centers_2d = pca.transform(km.cluster_centers_)
    anchor_emb = np.array(embed(list(anchors.values())), dtype=float)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    if k == 2:
        s = [[cos(km.cluster_centers_[ci], anchor_emb[ai]) for ai in range(2)] for ci in range(2)]
        order = (0, 1) if s[0][0] + s[1][1] >= s[0][1] + s[1][0] else (1, 0)
        centroids = {names[order[ci]]: centers_2d[ci].tolist() for ci in range(2)}
    else:
        centroids = {names[ci]: centers_2d[ci].tolist() for ci in range(k)}

    # Calibrate the chat/agentic ramp to the corpus distance distribution so the
    # boundary matches the real projection scale (fixed 0.3/1.0 constants were
    # tuned for a unit-scale dev projection and never fired on real embeddings).
    cen = np.array(list(centroids.values()))
    dists = np.min(np.linalg.norm(coords[:, None, :] - cen[None, :, :], axis=2), axis=1)
    dist_near = float(np.percentile(dists, 40))
    dist_far = float(np.percentile(dists, 90))

    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/pca_model.json").write_text(json.dumps({
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "embedding_model": MODEL,
        "dim": dim,
    }, indent=2))
    Path(f"{output_dir}/pca_centroids.json").write_text(json.dumps({
        "centroids": centroids,
        "dist_near": dist_near,
        "dist_far": dist_far,
    }, indent=2))
    Path(f"{output_dir}/pca_labels.json").write_text(json.dumps([
        {"id": items[i]["id"], "x": float(coords[i, 0]), "y": float(coords[i, 1])}
        for i in range(len(items))
    ], indent=2))
    print(f"Saved pca_model.json (dim={dim}), pca_centroids.json (dist_near={dist_near:.3f}, dist_far={dist_far:.3f}), pca_labels.json to {output_dir}/")
    print(f"Centroids (corpus k-means, anchor-labeled): {centroids}")
    print("Set EMBEDDINGS_API_KEY (same provider) on the backend so runtime embeddings match this model.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--papers", default="data/papers.json")
    p.add_argument("--graph", default="data/graph.json")
    p.add_argument("--output-dir", default="data")
    args = p.parse_args()
    fit_pca(args.papers, args.graph, args.output_dir)
