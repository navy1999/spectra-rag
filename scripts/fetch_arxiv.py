"""
Fetches arXiv papers for the spectra-rag knowledge graph.

Two pools, deliberately mixed:
  * FOUNDATIONAL_IDS — famous anchor papers (Transformer, BERT, ...). The PCA
    router uses these as semantic anchors, and they are the *in-distribution*
    entities a small model already knows cold.
  * RECENT papers — the most recently submitted papers across several arXiv
    categories. Their authors and titles are *out-of-distribution*: a small
    model has not memorized them, so retrieval (RAG), the trie entity guard
    (A2), and the SVD redundancy penalty (A3) finally have something to do.
    This is what makes the control-surface ablation measurable — on the
    foundational-only corpus the ON/OFF answers come out byte-identical.

Usage:
  python scripts/fetch_arxiv.py --max 200 --recent 180
  python scripts/fetch_arxiv.py --foundational-only      # just the anchors
"""
import json, argparse, os

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

# Pull *recent* (hence obscure) papers from a spread of subfields so authors are
# diverse and unlikely to be famous / memorized by a small model.
RECENT_CATEGORIES = [
    "cs.CL", "cs.LG", "cs.CV", "cs.AI", "cs.IR",
    "stat.ML", "cs.NE", "eess.AS", "cs.RO", "cs.SD",
]


def _paper_record(r):
    aid = r.entry_id.split("/")[-1].split("v")[0]
    return aid, {
        "id": aid,
        "title": " ".join(r.title.split()),
        "authors": [a.name for a in r.authors[:6]],
        "abstract": " ".join(r.summary.split())[:500],
        "year": r.published.year,
        "categories": r.categories[:3],
    }


def fetch_papers(output_path="data/papers.json", max_total=200, recent=180, foundational_only=False):
    try:
        import arxiv
    except ImportError:
        print("arxiv package not installed. Run: pip install arxiv")
        return

    papers = {}
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    print(f"Fetching {len(FOUNDATIONAL_IDS)} foundational anchor papers...")
    for arxiv_id in FOUNDATIONAL_IDS:
        try:
            results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
            if results:
                _, rec = _paper_record(results[0])
                rec["id"] = arxiv_id
                papers[arxiv_id] = rec
                print(f"  + {rec['title'][:60]}")
        except Exception as e:
            print(f"  ! {arxiv_id}: {e}")

    if not foundational_only and recent > 0:
        per_cat = max(1, recent // len(RECENT_CATEGORIES))
        print(f"\nFetching ~{per_cat} recent papers from each of {len(RECENT_CATEGORIES)} categories...")
        for cat in RECENT_CATEGORIES:
            if len(papers) >= max_total:
                break
            try:
                search = arxiv.Search(
                    query=f"cat:{cat}",
                    max_results=per_cat,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                )
                got = 0
                for r in client.results(search):
                    aid, rec = _paper_record(r)
                    if aid not in papers:
                        papers[aid] = rec
                        got += 1
                    if len(papers) >= max_total:
                        break
                print(f"  + {cat}: +{got} (total {len(papers)})")
            except Exception as e:
                print(f"  ! {cat}: {e}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(papers.values()), f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(papers)} papers to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/papers.json")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--recent", type=int, default=180, help="approx number of recent papers to add across categories")
    p.add_argument("--foundational-only", action="store_true", help="fetch only the famous anchor papers")
    args = p.parse_args()
    fetch_papers(args.output, args.max, args.recent, args.foundational_only)
