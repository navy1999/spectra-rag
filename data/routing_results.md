## A1 routing evaluation

40 questions · 20 labeled agentic / 20 chat. Routing accuracy = agreement with the label; agentic-rate is a cost proxy (agentic triggers retrieval + the vote ensemble).

| Router | Routing accuracy | Agentic-rate |
|---|---|---|
| pca16_lda | 95% | 50% |
| length | 52% | 42% |
| hit_count | 65% | 65% |
| always_agentic | 50% | 100% |
| always_chat | 50% | 0% |

A router earns its complexity only if it beats `length`/`hit_count` and the `always_*` baselines. If a one-liner ties the learned router, that is a real finding (simplify).
