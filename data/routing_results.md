## A1 routing evaluation

40 questions · 20 labeled agentic / 20 chat. Routing accuracy = agreement with the label; agentic-rate is a cost proxy (agentic triggers retrieval + the vote ensemble).

> WARNING: no real embeddings (no EMBEDDINGS_API_KEY or dimension mismatch) — the `pca` row routed on the dev sketch and is NOT meaningful.

| Router | Routing accuracy | Agentic-rate |
|---|---|---|
| pca16_lda | 42% | 48% |
| length | 52% | 42% |
| hit_count | 65% | 85% |
| always_agentic | 50% | 100% |
| always_chat | 50% | 0% |

A router earns its complexity only if it beats `length`/`hit_count` and the `always_*` baselines. If a one-liner ties the learned router, that is a real finding (simplify).
