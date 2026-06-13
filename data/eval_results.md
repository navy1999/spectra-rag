## Phase 1 evaluation

Model `openai/gpt-oss-20b:free` · scored on 22 of 22 questions · same model across conditions · temperature 0.30 · retrieval held fixed across both RAG conditions (seed + 1-hop BFS).

Generated 2026-06-13.

| Metric | raw | rag_plain | rag_spectra |
|---|---|---|---|
| Entity exact-spelling rate (higher better) | 73.5% | 79.2% | 81.4% |
| Entity near-miss rate (lower better) | 14.0% | 15.5% | 11.7% |
| Entity recall, any form (higher better) | 87.5% | 94.7% | 93.2% |
| Repetition: distinct-2 (higher better) | 0.913 | 0.922 | 0.939 |
| Groundedness, RAG only (higher better) | — | 82.2% | 87.5% |
| Trie corrections (total) | — | — | 5 |
| Mean latency (ms) | 5585 | 7733 | 962 |

_Entity fidelity is the trie guard (A2): the spectra column should raise exact-spelling and cut near-misses. Repetition (distinct-2) is the SVD penalty (A3). Metrics are judge-free string measures; routing (A1) and the vote evaluator (A4) are reported separately._
