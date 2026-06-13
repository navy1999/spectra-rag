## A1 routing evaluation

16 questions · 8 labeled agentic / 8 chat. Routing accuracy = agreement with the label; agentic-rate is a cost proxy (agentic triggers retrieval + the vote ensemble).

| Router | Routing accuracy | Agentic-rate |
|---|---|---|
| pca | 69% | 56% |
| length | 75% | 25% |
| hit_count | 69% | 56% |
| always_agentic | 50% | 100% |
| always_chat | 50% | 0% |

A router earns its complexity only if it beats `length`/`hit_count` and the `always_*` baselines. If a one-liner ties `pca`, that is a real finding (simplify).
