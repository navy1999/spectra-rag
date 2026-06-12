"""
Builds data/graph.json from data/papers.json.
Extracts Paper/Author/Topic/Institution nodes and edges.
"""
import json, re, argparse, os
from pathlib import Path
from collections import defaultdict

KNOWN_AFFILIATIONS = {
    "Ashish Vaswani": "Google Brain", "Noam Shazeer": "Google Brain",
    "Jacob Devlin": "Google", "Ming-Wei Chang": "Google",
    "Tom Brown": "OpenAI", "Sam Altman": "OpenAI",
    "Edward Hu": "Microsoft Research", "Yelong Shen": "Microsoft Research",
    "Tri Dao": "Stanford University", "Christopher Ré": "Stanford University",
    "Hugo Touvron": "Meta AI", "Thibaut Lavril": "Meta AI",
}

TOPIC_KEYWORDS = {
    "Transformer Architecture": ["transformer", "attention", "self-attention", "multi-head"],
    "Pre-training": ["pre-train", "pretraining", "masked language", "next sentence"],
    "Parameter-Efficient Fine-Tuning": ["lora", "adapter", "fine-tun", "peft"],
    "Efficient Attention": ["flash", "efficient attention", "linear attention", "sparse attention"],
    "Reinforcement Learning": ["rlhf", "reward model", "reinforcement", "human feedback"],
    "Retrieval Augmented Generation": ["retrieval", "rag", "knowledge graph", "vector"],
    "Mixture of Experts": ["mixture of experts", "moe", "sparse mixture"],
    "In-Context Learning": ["in-context", "few-shot", "zero-shot", "prompt"],
}


def build_graph(papers_path: str = "data/papers.json", output_path: str = "data/graph.json"):
    papers = json.loads(Path(papers_path).read_text()) if Path(papers_path).exists() else []
    if not papers:
        print("No papers found, using default graph")
        return

    nodes, edges = [], []
    seen_authors, seen_topics, seen_insts = {}, {}, {}

    for paper in papers:
        pid = f"p_{paper['id'].replace('.', '_')}"
        nodes.append({
            "id": pid,
            "type": "paper",
            "name": paper["title"],
            "props": {"year": paper.get("year"), "arxiv": paper.get("id"), "abstract": paper.get("abstract", "")[:200]},
        })

        # Authors
        for author in paper.get("authors", []):
            if author not in seen_authors:
                aid = f"a_{len(seen_authors)}"
                seen_authors[author] = aid
                nodes.append({"id": aid, "type": "author", "name": author, "props": {}})
                inst = KNOWN_AFFILIATIONS.get(author)
                if inst:
                    if inst not in seen_insts:
                        iid = f"i_{len(seen_insts)}"
                        seen_insts[inst] = iid
                        nodes.append({"id": iid, "type": "institution", "name": inst, "props": {}})
                    edges.append({"from": aid, "to": seen_insts[inst], "rel": "affiliated"})
            edges.append({"from": seen_authors[author], "to": pid, "rel": "authored"})

        # Topics
        text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                if topic not in seen_topics:
                    tid = f"t_{len(seen_topics)}"
                    seen_topics[topic] = tid
                    nodes.append({"id": tid, "type": "topic", "name": topic, "props": {}})
                edges.append({"from": pid, "to": seen_topics[topic], "rel": "about"})

        # Citation edges via arXiv ID regex
        for cited_id in re.findall(r'\b\d{4}\.\d{4,5}\b', paper.get("abstract", "")):
            edges.append({"from": pid, "to": f"p_{cited_id.replace('.', '_')}", "rel": "cites"})

    graph = {"nodes": nodes, "edges": edges}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Path(output_path).write_text(json.dumps(graph, indent=2))
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--papers", default="data/papers.json")
    p.add_argument("--output", default="data/graph.json")
    p.parse_args()
    args = p.parse_args()
    build_graph(args.papers, args.output)
