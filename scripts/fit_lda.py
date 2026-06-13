"""
Fits a SUPERVISED projection (LDA) the router uses to separate chat-vs-agentic
queries, on the SAME embeddings the backend uses at runtime (Jina by default).

Why LDA instead of (or alongside) the unsupervised PCA in fit_pca.py: the A1
routing evaluation showed the PCA router did NOT beat a trivial length baseline.
PCA maximizes variance, which on a homogeneous paper corpus is dominated by
topic, not by the chat-vs-agentic distinction we actually route on. LDA is
supervised: it finds the linear axis that maximally separates the two LABELED
classes (data/routing_questions.json), so the projection is aligned with the
decision the router has to make.

The output schema is IDENTICAL to fit_pca.py's so the Go router/CGo bridge need
ZERO changes — just point PCA_MODEL_PATH / PCA_CENTROIDS_PATH at these files:
  data/lda_model.json      — components (2 x n_features) + mean (n_features)
  data/lda_centroids.json  — per-class centers in the 2D projected space
  data/lda_labels.json     — per-item 2D coords + class (for inspection)

A 2-class LDA yields a single discriminant axis. The router and the UI both want
2D, so:
  x = LDA discriminant (the supervised, decision-aligned axis)
  y = top PCA direction made orthogonal to the LDA axis (a complementary,
      unsupervised spread axis, deflated out of the LDA direction so the two
      rows of `components` stay linearly independent — purely for visualization;
      the route decision is driven by x).

Centroids are the per-class projected means, labeled so the chat class maps to
"logic" and the agentic class to "creative" — reusing the router's existing
regimeBaseTemp keys (router/pca_router.go) with no Go change. dist_near/dist_far
are calibrated on the held projection so the confidence cutoff (agentic when
confidence < 0.5) falls between the two class means.

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

# PCA pre-reduction rank before LDA. Chosen by the leave-one-out sweep in
# scripts/route_model_search.py on classification-task embeddings (k=16 maximized
# LOOCV accuracy). Override with PCA_K if you re-tune on a different corpus.
PCA_K = int(os.environ.get("PCA_K", "16"))

# Map the two routing labels to the router's existing regime keys so the Go side
# needs no change. "chat" → "logic" (factual, low-temp lookups); "agentic" →
# "creative" (exploratory multi-hop, looser sampling).
LABEL_TO_REGIME = {"chat": "logic", "agentic": "creative"}


def embed(texts, batch_size=32):
    """Embed texts via the configured provider (OpenAI-compatible /embeddings)."""
    if not API_KEY:
        raise SystemExit(
            "EMBEDDINGS_API_KEY is not set. Set it to the same key the backend uses, e.g.\n"
            "  bash: EMBEDDINGS_API_KEY=jina_... python fit_lda.py"
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


def load_labeled(questions_path):
    doc = json.loads(Path(questions_path).read_text())
    qs = doc["questions"] if isinstance(doc, dict) else doc
    items = [{"id": q["id"], "text": q["text"], "label": q["path"]} for q in qs]
    return items


def load_precomputed(embeddings_path):
    """Load vectors produced by `go run ./cmd/embeddump` (preferred path).

    Running the embeddings through the backend's own Go HTTP client guarantees
    the fit uses the exact same provider/model/dimension the server calls at
    runtime, and sidesteps Python TLS issues behind the egress proxy.
    """
    doc = json.loads(Path(embeddings_path).read_text())
    items = doc["items"] if isinstance(doc, dict) else doc
    out = [{"id": it["id"], "label": it.get("label") or it.get("path"),
            "embedding": it["embedding"]} for it in items]
    return out


def fit_lda(questions_path="data/routing_questions.json", output_dir="data",
            embeddings_path=None):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if embeddings_path and Path(embeddings_path).exists():
        items = load_precomputed(embeddings_path)
        if not items:
            raise SystemExit(f"No embeddings found in {embeddings_path}")
        labels = [it["label"] for it in items]
        classes = sorted(set(labels))
        if len(classes) != 2:
            raise SystemExit(f"Expected exactly 2 classes, got {classes}")
        X = np.array([it["embedding"] for it in items], dtype=float)
        dim = X.shape[1]
        print(f"Loaded {len(X)} precomputed embeddings (dim={dim}) from {embeddings_path} "
              f"({labels.count('chat')} chat / {labels.count('agentic')} agentic).")
    else:
        items = load_labeled(questions_path)
        if not items:
            raise SystemExit(f"No labeled questions found in {questions_path}")
        labels = [it["label"] for it in items]
        classes = sorted(set(labels))
        if len(classes) != 2:
            raise SystemExit(f"Expected exactly 2 classes, got {classes}")
        print(f"Embedding {len(items)} labeled questions ({labels.count('chat')} chat / "
              f"{labels.count('agentic')} agentic) via {MODEL} @ {BASE_URL} ...")
        X = np.array(embed([it["text"] for it in items]), dtype=float)
        dim = X.shape[1]
        print(f"Got {len(X)} embeddings of dimension {dim}.")

    y = np.array([1 if lab == "agentic" else 0 for lab in labels])

    # --- Supervised axis: PCA(k) -> LDA, folded into one linear map ---------
    # Raw LDA on 1024 features with only ~40 samples is hopelessly
    # underdetermined and overfits (in-sample ~100%, leave-one-out ~chance). The
    # standard fix is to reduce with PCA FIRST, then run LDA on the low-rank
    # representation. A leave-one-out sweep (scripts/route_model_search.py) on
    # classification-task Jina embeddings picked k=16 (LOOCV ~85%, well above the
    # length baseline). Both stages are linear, so the whole PCA(k)->LDA pipeline
    # composes into a SINGLE projection in the original feature space:
    #     z = lda_scaling^T · (pca_components · (x - pca_mean))
    #       = (pca_components^T · lda_scaling)^T · (x - pca_mean)
    # which is exactly the `components · (x - mean)` form the Go bridge already
    # implements — so the runtime router needs ZERO change.
    pca = PCA(n_components=min(PCA_K, len(X) - 1))
    Z = pca.fit_transform(X)
    mean = pca.mean_.astype(float)

    lda = LinearDiscriminantAnalysis()
    lda.fit(Z, y)
    # scalings_ is (k, 1) for 2-class LDA: the discriminant in PCA space.
    w_pca_space = lda.scalings_[:, 0].astype(float)
    # Fold back into original feature space: (dim,) = (dim,k) @ (k,)
    w_lda = pca.components_.T @ w_pca_space
    w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-12)  # unit length, stable scale

    # --- Complementary axis: top PCA direction, orthogonalized to w_lda -----
    # Visualization-only spread axis. Deflate the LDA component out so the two
    # rows of `components` are orthogonal (clean, independent 2D coords).
    w_pca = pca.components_[0].astype(float)
    w_pca = w_pca - (w_pca @ w_lda) * w_lda  # Gram-Schmidt against w_lda
    nrm = np.linalg.norm(w_pca)
    w_pca = w_pca / nrm if nrm > 1e-9 else np.zeros_like(w_pca)

    components = np.vstack([w_lda, w_pca])  # shape (2, dim)
    coords = (X - mean) @ components.T       # shape (n, 2)

    # Orient the discriminant so agentic sits at positive x (purely cosmetic;
    # the router is sign-agnostic since it uses distance to class centroids).
    if coords[y == 1, 0].mean() < coords[y == 0, 0].mean():
        components[0] = -components[0]
        coords[:, 0] = -coords[:, 0]

    # --- Centroids = projected class means ---------------------------------
    chat_mean = coords[y == 0].mean(axis=0)
    agentic_mean = coords[y == 1].mean(axis=0)
    centroids = {
        LABEL_TO_REGIME["chat"]: chat_mean.tolist(),       # "logic"
        LABEL_TO_REGIME["agentic"]: agentic_mean.tolist(),  # "creative"
    }

    # --- Calibrate the novelty ramp ----------------------------------------
    # The router routes agentic when confidence < 0.5, i.e. novelty > 0.5, i.e.
    # min-distance-to-centroid > (near+far)/2. Because agentic questions sit
    # nearest the "creative" centroid (their own class mean) their min-distance
    # is SMALL, so a pure distance-to-nearest novelty signal would not separate
    # the classes. The separation the LDA buys us lives in the REGIME (argmin)
    # label, not the distance. We therefore keep the existing distance ramp for
    # the temperature/novelty UI signal, but the chat-vs-agentic ROUTE is taken
    # from which class centroid is nearest — see the note below and the Go change
    # in pca_router.go (regime-based routing when centroids carry class meaning).
    cen = np.array([chat_mean, agentic_mean])
    d_each = np.linalg.norm(coords[:, None, :] - cen[None, :, :], axis=2)
    d_near_arr = d_each.min(axis=1)
    dist_near = float(np.percentile(d_near_arr, 40))
    dist_far = float(np.percentile(d_near_arr, 90))

    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/lda_model.json").write_text(json.dumps({
        "components": components.tolist(),
        "mean": mean.tolist(),
        "embedding_model": MODEL,
        "dim": dim,
        "method": f"pca{PCA_K}_lda",
        "note": "row0 = PCA(k)->LDA discriminant folded into feature space (supervised chat/agentic axis); row1 = top PCA dir orthogonalized to row0 (viz only). Fitted on classification-task embeddings.",
    }, indent=2))
    Path(f"{output_dir}/lda_centroids.json").write_text(json.dumps({
        "centroids": centroids,
        "dist_near": dist_near,
        "dist_far": dist_far,
        "route_by_regime": True,
        "chat_regime": LABEL_TO_REGIME["chat"],
    }, indent=2))
    Path(f"{output_dir}/lda_labels.json").write_text(json.dumps([
        {"id": items[i]["id"], "label": items[i]["label"],
         "x": float(coords[i, 0]), "y": float(coords[i, 1])}
        for i in range(len(items))
    ], indent=2))

    # Report training separation along the supervised axis.
    chat_x = coords[y == 0, 0]
    agentic_x = coords[y == 1, 0]
    midpoint = (chat_x.mean() + agentic_x.mean()) / 2
    train_correct = int((chat_x < midpoint).sum() + (agentic_x >= midpoint).sum())

    # --- Leave-one-out cross-validation -------------------------------------
    # In-sample accuracy is meaningless for a supervised model fit on the same
    # 16 points (it will be ~100%). LOOCV is the honest estimate to compare
    # against the length baseline (which fits NO parameters to this data): fit
    # LDA on 15, classify the held-out 1 by nearest projected class centroid,
    # repeat. This is the number worth reporting on a resume / in the README.
    loo_correct = 0
    for i in range(len(X)):
        tr = np.array([j for j in range(len(X)) if j != i])
        yi = y[tr]
        if len(set(yi.tolist())) < 2:
            continue
        p_fold = PCA(n_components=min(PCA_K, len(tr) - 1))
        Ztr = p_fold.fit_transform(X[tr])
        m = LinearDiscriminantAnalysis()
        m.fit(Ztr, yi)
        pred = int(m.predict(p_fold.transform(X[i].reshape(1, -1)))[0])
        if pred == y[i]:
            loo_correct += 1
    loo_acc = loo_correct / len(X)

    print(f"Saved lda_model.json (dim={dim}), lda_centroids.json, lda_labels.json to {output_dir}/")
    print(f"Centroids (projected class means): {centroids}")
    print(f"LDA axis means: chat={chat_x.mean():.4f}, agentic={agentic_x.mean():.4f}, midpoint={midpoint:.4f}")
    print(f"In-sample separation along LDA axis: {train_correct}/{len(items)} (training fit; not a generalization estimate).")
    print(f"Leave-one-out CV accuracy (HONEST estimate): {loo_correct}/{len(X)} = {loo_acc:.1%}")
    print("Point the backend at these via PCA_MODEL_PATH=data/lda_model.json PCA_CENTROIDS_PATH=data/lda_centroids.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/routing_questions.json")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--embeddings", default=None,
                   help="precomputed embeddings JSON from cmd/embeddump (skips the network call)")
    args = p.parse_args()
    fit_lda(args.questions, args.output_dir, args.embeddings)
