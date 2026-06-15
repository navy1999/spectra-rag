## Phase 1 evaluation

Model `liquid/lfm-2.5-1.2b-instruct:free` · scored on raw 21 / plain 21 / spectra 22 of 22 questions (remainder skipped: provider rate-limited) · same model across conditions · temperature 0.30 · retrieval held fixed across both RAG conditions (seed + 1-hop BFS).

Generated 2026-06-14.

| Metric | raw | rag_plain | rag_spectra |
|---|---|---|---|
| Entity exact-spelling rate (higher better) | 78.2% | 80.2% | 81.1% |
| Entity near-miss rate (lower better) | 2.8% | 13.1% | 12.5% |
| Entity recall, any form (higher better) | 81.0% | 93.3% | 93.6% |
| Repetition: distinct-2 (higher better) | 0.932 | 0.936 | 0.938 |
| Groundedness, RAG only (higher better) | — | 100.0% | 100.0% |
| Trie corrections (total) | — | — | 1 |
| Mean latency (ms) | 1355 | 2058 | 102 |

_Entity fidelity is the trie guard (A2): the spectra column should raise exact-spelling and cut near-misses. Repetition (distinct-2) is the SVD penalty (A3). Metrics are judge-free string measures; routing (A1) and the vote evaluator (A4) are reported separately._
