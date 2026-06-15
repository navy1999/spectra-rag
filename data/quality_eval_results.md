## Phase 2 — end-to-end answer quality (LLM-as-judge)

Generator `liquid/lfm-2.5-1.2b-instruct:free` · judge `openai/gpt-oss-120b:free` · scored on 22 of 22 questions.
Blind pairwise: **ON = `rag_spectra`** vs **OFF = `rag_plain`**, both A/B orderings per pair (a side scores only if it wins consistently, cancelling position bias).

Generated 2026-06-14.

| Outcome | count |
|---|---|
| ON (`rag_spectra`) preferred | 0 |
| OFF (`rag_plain`) preferred | 1 |
| Tie / inconsistent across orderings | 21 |

**ON win-rate among decisive comparisons: 0/1 = 0.0%** (95% CI 0.0%–79.3%).

**Inconclusive at this N:** the 95% CI spans 50%, so this run does not establish that the control surfaces change end-to-end answer quality.

_LLM-as-judge is a proxy for human preference, not ground truth. The judge is a different, larger model than the one under test (no self-grading), comparisons are blind, and position bias is controlled. N is small and questions are in-distribution, so treat this as directional, not definitive._
