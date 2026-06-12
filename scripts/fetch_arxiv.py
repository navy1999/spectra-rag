"""
Fetches foundational ML/NLP papers from arXiv and saves to data/papers.json.
Usage: python scripts/fetch_arxiv.py [--output data/papers.json]
"""
import json, time, argparse, os
from pathlib import Path

FOUNDATIONAL_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3
    "2106.09685",  # LoRA
    "2205.14135",  # FlashAttention
    "1301.3666",   # Word2Vec
    "1607.06450",  # Layer Norm
    "1512.03385",  # ResNet
    "2010.11929",  # ViT
    "2203.15556",  # InstructGPT
    "2302.13971",  # LLaMA
    "2310.06825",  # Mistral
    "2307.09288",  # LLaMA 2
    "2210.11610",  # Flan-T5
    "2112.10752",  # RLHF
    "1910.10683",  # T5
    "2103.00020",  # CLIP
    "2204.05149",  # PaLM
    "1901.00596",  # XLNet
    "2108.12409",  # CodeX
]

SEARCH_QUERIES = [
    "large language model alignment",
    "retrieval augmented generation",
    "chain of thought prompting",
    "mixture of experts transformer",
    "state space models language",
]


def fetch_papers(output_path: str = "data/papers.json", max_total: int = 50):
    try:
        import arxiv
    except ImportError:
        print("arxiv package not installed. Run: pip install arxiv")
        return

    papers = {}

    # Fetch foundational by ID
    print(f"Fetching {len(FOUNDATIONAL_IDS)} foundational papers...")
    client = arxiv.Client()
    for arxiv_id in FOUNDATIONAL_IDS:
        try:
            results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
            if results:
                r = results[0]
                papers[arxiv_id] = {
                    "id": arxiv_id,
                    "title": r.title,
                    "authors": [a.name for a in r.authors[:5]],
                    "abstract": r.summary[:500],
                    "year": r.published.year,
                    "categories": r.categories[:3],
                }
                print(f"  ✓ {r.title[:60]}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ {arxiv_id}: {e}")

    # Fill to max_total via search
    remaining = max_total - len(papers)
    if remaining > 0:
        per_query = max(1, remaining // len(SEARCH_QUERIES))
        print(f"\nSearching for {remaining} more papers...")
        for query in SEARCH_QUERIES:
            try:
                results = list(client.results(arxiv.Search(query=query, max_results=per_query)))
                for r in results:
                    aid = r.entry_id.split("/")[-1].split("v")[0]
                    if aid not in papers:
                        papers[aid] = {
                            "id": aid,
                            "title": r.title,
                            "authors": [a.name for a in r.authors[:5]],
                            "abstract": r.summary[:500],
                            "year": r.published.year,
                            "categories": r.categories[:3],
                        }
                time.sleep(1)
            except Exception as e:
                print(f"  Search error: {e}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(papers.values()), f, indent=2)
    print(f"\nSaved {len(papers)} papers to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/papers.json")
    p.add_argument("--max", type=int, default=50)
    args = p.parse_args()
    fetch_papers(args.output, args.max)
