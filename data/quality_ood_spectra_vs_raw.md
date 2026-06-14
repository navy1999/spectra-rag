## Phase 2 — end-to-end answer quality (LLM-as-judge)

Generator `liquid/lfm-2.5-1.2b-instruct:free` · judge `openai/gpt-oss-120b:free` · scored on 30 of 30 questions.
Blind pairwise: **ON = `rag_spectra`** vs **OFF = `raw`**, both A/B orderings per pair (a side scores only if it wins consistently, cancelling position bias).

Generated 2026-06-14.

| Outcome | count |
|---|---|
| ON (`rag_spectra`) preferred | 21 |
| OFF (`raw`) preferred | 3 |
| Tie / inconsistent across orderings | 6 |

**ON win-rate among decisive comparisons: 21/24 = 87.5%** (95% CI 69.0%–95.7%).

**The control surfaces help:** the lower 95% bound stays above 50%, so ON is preferred more often than chance.

_LLM-as-judge is a proxy for human preference, not ground truth. The judge is a different, larger model than the one under test (no self-grading), comparisons are blind, and position bias is controlled. N is small and questions are in-distribution, so treat this as directional, not definitive._
