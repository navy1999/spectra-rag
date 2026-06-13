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

    # Regime centroids via SEMANTIC ANCHORS, not keyword buckets. We embed a
    # natural-language description of each regime with the same model and project
    # it through the fitted PCA; a query then routes to whichever description it
    # lands nearest in PCA space. This grounds the regime in meaning instead of a
    # brittle keyword list, and needs no labeled data.
    #
    # Names MUST match the backend's regimeBaseTemp keys (router/pca_router.go).
    # Add regimes by adding entries here and a base temperature there.
    #
    # Honest limitation: the PCA basis is still fit on the corpus (unsupervised),
    # so its axes reflect corpus variance, not the logic/creative axis — anchors
    # fix the *labels*, not the *projection*. Fitting LDA on labeled example
    # queries would also fix the axis (see the A1 notes).
    anchors = {
        "logic": "A precise, factual, analytical question seeking a correct, deterministic, lookup-style answer grounded in specific facts.",
        "creative": "An open-ended, generative, exploratory or imaginative request inviting free-form, speculative, or creative writing.",
    }
    anchor_coords = pca.transform(embed(list(anchors.values())))
    centroids = {name: anchor_coords[i].tolist() for i, name in enumerate(anchors)}

    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/pca_model.json").write_text(json.dumps({
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "embedding_model": MODEL,
        "dim": dim,
    }, indent=2))
    Path(f"{output_dir}/pca_centroids.json").write_text(json.dumps(centroids, indent=2))
    Path(f"{output_dir}/pca_labels.json").write_text(json.dumps([
        {"id": items[i]["id"], "x": float(coords[i, 0]), "y": float(coords[i, 1])}
        for i in range(len(items))
    ], indent=2))
    print(f"Saved pca_model.json (dim={dim}), pca_centroids.json, pca_labels.json to {output_dir}/")
    print("Set EMBEDDINGS_API_KEY (same provider) on the backend so runtime embeddings match this model.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--papers", default="data/papers.json")
    p.add_argument("--graph", default="data/graph.json")
    p.add_argument("--output-dir", default="data")
    args = p.parse_args()
    fit_pca(args.papers, args.graph, args.output_dir)
