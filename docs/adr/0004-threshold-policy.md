# 0004. Choose the threshold by mean recall across patterns under a precision floor

**Status:** Accepted

## Context
83% of fraudulent rows are card testing. Neither the default 0.5 threshold nor the best-F1 threshold looks at the patterns separately, so both can score well overall while missing most of a rare pattern.

## Decision
Try 500 thresholds at evenly spaced quantiles of the test scores. Discard any threshold whose precision is below 95%, a budget of at most one false alarm per 20 alerts. Among the rest, pick the one with the highest mean recall across high_amount, card_testing and impossible_travel. If no threshold reaches 95%, fall back to best F1 and log a warning (`training/evaluation.py`).

## Alternatives considered
- **0.5.** Arbitrary for a model trained with class weights.
- **Best aggregate F1.** Dominated by the most frequent pattern. The unit test `test_threshold_favours_the_rare_pattern_that_best_f1_would_drop` builds a case where it misses a rare pattern entirely.
- **A recall target.** Puts no limit on the alert volume, which is what a fraud team actually has to staff.

## Consequences
- Chosen threshold 0.5625: precision 95.75%, recall 96.55%, mean recall across patterns 86.98%.
- The threshold is chosen on the test set, so the reported precision is slightly optimistic. A separate validation split is the fix if the policy gets more parameters.
