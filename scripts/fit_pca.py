"""
Fits PCA on paper embeddings. Saves:
  data/pca_model.json      — components + mean (for C++ engine)
  data/pca_centroids.json  — logic/creative cluster centers
  data/pca_labels.json     — per-paper 2D coords
"""
import json, argparse
from pathlib import Path

def fit_pca(papers_path="data/papers.json", output_dir="data"):
    papers = json.loads(Path(papers_path).read_text()) if Path(papers_path).exists() else []
    if not papers:
        print("No papers to embed, using default centroids")
        Path(f"{output_dir}/pca_centroids.json").write_text(json.dumps({"logic": [0.42, -0.18], "creative": [-0.31, 0.29]}, indent=2))
        return

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
        import numpy as np
    except ImportError:
        print("Run: pip install sentence-transformers scikit-learn numpy")
        return

    print("Loading sentence-transformers...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = []
    for p in papers:
        # Title weight ×3
        texts.append((p.get("title", "") + " ") * 3 + p.get("abstract", ""))

    print(f"Embedding {len(texts)} papers...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    # Classify as logic (BERT/GPT/LoRA) vs creative (generative/art)
    # Simple heuristic: first component separates factual vs generative
    logic_mask = [i for i, p in enumerate(papers) if any(
        kw in (p.get("title","")+" "+p.get("abstract","")).lower()
        for kw in ["bert", "classification", "question answering", "named entity", "fine-tun"]
    )]
    creative_mask = [i for i, p in enumerate(papers) if any(
        kw in (p.get("title","")+" "+p.get("abstract","")).lower()
        for kw in ["generate", "generation", "creative", "story", "image", "diffusion"]
    )]

    logic_center = coords[logic_mask].mean(axis=0).tolist() if logic_mask else [0.42, -0.18]
    creative_center = coords[creative_mask].mean(axis=0).tolist() if creative_mask else [-0.31, 0.29]

    pca_model = {
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/pca_model.json").write_text(json.dumps(pca_model, indent=2))
    Path(f"{output_dir}/pca_centroids.json").write_text(json.dumps({"logic": logic_center, "creative": creative_center}, indent=2))
    Path(f"{output_dir}/pca_labels.json").write_text(json.dumps([
        {"id": papers[i].get("id"), "title": papers[i].get("title"), "x": float(coords[i,0]), "y": float(coords[i,1])}
        for i in range(len(papers))
    ], indent=2))
    print(f"Saved PCA model + centroids to {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--papers", default="data/papers.json")
    p.add_argument("--output-dir", default="data")
    args = p.parse_args()
    fit_pca(args.papers, args.output_dir)
