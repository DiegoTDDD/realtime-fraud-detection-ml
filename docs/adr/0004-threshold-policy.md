# 0004. Choose the threshold by mean recall across patterns under a precision floor

**Status:** Accepted; since [ADR 0007](0007-validation-split-for-the-threshold.md) the threshold is chosen on the validation period

## Context
83% of fraudulent rows are card testing. Neither the default 0.5 threshold nor the best-F1 threshold looks at the patterns separately, so both can score well overall while missing most of a rare pattern.

## Decision
Try 500 thresholds at evenly spaced quantiles of the validation scores. Discard any threshold whose precision is below 95%, a budget of at most one false alarm per 20 alerts. Among the rest, pick the one with the highest mean recall across high_amount, card_testing and impossible_travel. If no threshold reaches 95%, fall back to best F1 and log a warning (`training/evaluation.py`).

## Alternatives considered
- **0.5.** Arbitrary for a model trained with class weights.
- **Best aggregate F1.** Dominated by the most frequent pattern. The unit test `test_threshold_favours_the_rare_pattern_that_best_f1_would_drop` builds a case where it misses a rare pattern entirely.
- **A recall target.** Puts no limit on the alert volume, which is what a fraud team actually has to staff.

## Consequences
- Chosen threshold 0.578 on the validation period (precision 96.11%, recall 97.30%). On the test period, scored once: precision 96.60%, recall 96.60%, mean recall across patterns 87.17%.
- Before ADR 0007 the threshold was chosen on the test set (0.5625, precision 95.75%), which made the reported precision slightly optimistic; ADR 0007 is that fix.
