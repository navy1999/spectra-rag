"""
Full ingestion pipeline: fetch → build graph → fit PCA
Usage: python scripts/ingest.py [--skip-fetch] [--output-dir data]
"""
import argparse, subprocess, sys, os
from pathlib import Path

REQUIRED_OUTPUTS = [
    "data/papers.json",
    "data/graph.json",
    "data/pca_model.json",
    "data/pca_centroids.json",
    "data/pca_labels.json",
]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(description="Run the full spectra-rag ingestion pipeline")
    p.add_argument("--skip-fetch", action="store_true", help="Skip arXiv fetch (use existing papers.json)")
    p.add_argument("--output-dir", default="data")
    args = p.parse_args()

    py = sys.executable
    scripts = Path(__file__).parent

    if not args.skip_fetch:
        run([py, str(scripts / "fetch_arxiv.py"), "--output", f"{args.output_dir}/papers.json"])
    else:
        print("Skipping fetch (--skip-fetch)")

    run([py, str(scripts / "build_graph.py"), "--papers", f"{args.output_dir}/papers.json",
         "--output", f"{args.output_dir}/graph.json"])
    run([py, str(scripts / "fit_pca.py"), "--papers", f"{args.output_dir}/papers.json",
         "--output-dir", args.output_dir])
    if os.environ.get("EMBEDDINGS_API_KEY"):
        run([py, str(scripts / "embed_nodes.py"), "--graph", f"{args.output_dir}/graph.json",
             "--out", f"{args.output_dir}/node_embeddings.json"])
    else:
        print("Skipping node embeddings (EMBEDDINGS_API_KEY unset); retrieval stays lexical-only.")

    # Regenerate the out-of-domain eval question set from the fresh graph so the
    # control-surface ablation (A2/A3) targets entities the model has not seen.
    run([py, str(scripts / "make_eval_questions.py"), "--graph", f"{args.output_dir}/graph.json",
         "--out", f"{args.output_dir}/eval_questions_ood.json"])

    print("\n=== Validation ===")
    missing = [f for f in REQUIRED_OUTPUTS if not Path(f).exists()]
    if missing:
        print(f"MISSING: {missing}")
        sys.exit(1)
    print("All required outputs present ✓")


if __name__ == "__main__":
    main()
