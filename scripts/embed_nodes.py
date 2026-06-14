"""
Embeds every knowledge-graph node (name + abstract) with the SAME provider the
backend uses at query time, so the agent loop can seed retrieval semantically:
the query vector and the node vectors live in one space. Writes
data/node_embeddings.json: {"model":..., "dim":N, "embeddings": {nodeID: [...]}}.

The backend loads this file if present and falls back to lexical-only seeding if
not, so it's optional — but with a few-hundred-node graph, semantic seeding is
where retrieval quality comes from. Re-run this whenever the graph changes.

Requires the embeddings key, matching the backend:
  EMBEDDINGS_API_KEY   (required)
  EMBEDDINGS_BASE_URL  (default https://api.jina.ai/v1)
  EMBEDDINGS_MODEL     (default jina-embeddings-v3)
  EMBEDDINGS_TASK      (default "classification" — MUST match the backend's
                        EMBEDDINGS_TASK: the live query vector is embedded once
                        with that task and reused for both routing and semantic
                        seeding, so node embeddings have to share the same space.
                        Consistency over retrieval-optimality; a fully-optimal
                        setup would embed the query twice with different adapters.)
"""
import argparse
import json
import os
from pathlib import Path

import requests

BASE_URL = os.environ.get("EMBEDDINGS_BASE_URL", "https://api.jina.ai/v1")
MODEL = os.environ.get("EMBEDDINGS_MODEL", "jina-embeddings-v3")
API_KEY = os.environ.get("EMBEDDINGS_API_KEY", "")
TASK = os.environ.get("EMBEDDINGS_TASK", "classification")


def embed(texts, batch=32):
    if not API_KEY:
        raise SystemExit(
            "EMBEDDINGS_API_KEY is not set. Use the same key the backend uses, e.g.\n"
            '  PowerShell:  $env:EMBEDDINGS_API_KEY="jina_..."; python embed_nodes.py'
        )
    out = []
    for i in range(0, len(texts), batch):
        payload = {"model": MODEL, "input": texts[i : i + batch]}
        if TASK:
            payload["task"] = TASK
        r = requests.post(
            f"{BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        out.extend(item["embedding"] for item in r.json()["data"])
    return out


def node_text(n):
    text = n.get("name", "")
    props = n.get("props") or {}
    if props.get("abstract"):
        text += ". " + props["abstract"]
    return text


def main(graph_path, out_path):
    graph = json.loads(Path(graph_path).read_text())
    nodes = graph.get("nodes", [])
    if not nodes:
        raise SystemExit(f"no nodes in {graph_path}")
    print(f"Embedding {len(nodes)} nodes via {MODEL} @ {BASE_URL} ...")
    vecs = embed([node_text(n) for n in nodes])
    dim = len(vecs[0]) if vecs else 0
    embeddings = {nodes[i]["id"]: [round(float(x), 6) for x in vecs[i]] for i in range(len(nodes))}
    Path(out_path).write_text(json.dumps({"model": MODEL, "dim": dim, "embeddings": embeddings}))
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved {len(embeddings)} node embeddings (dim={dim}, {size_mb:.1f} MB) to {out_path}")
    print("Commit this file so the deployed backend gets semantic seed retrieval.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="data/graph.json")
    p.add_argument("--out", default="data/node_embeddings.json")
    args = p.parse_args()
    main(args.graph, args.out)
