# 0007. Choose the threshold on a validation period; report the test period once

**Status:** Accepted (replaces the 80/20 split of ADR 0006)

## Context
The threshold policy (ADR 0004) searches 500 candidate thresholds. With an 80/20 split it searched them on the test set, so the reported precision and recall were the best of 500 tries on the same rows: slightly optimistic, and the README said so.

## Decision
Split by time into three consecutive periods: the earliest 60% train the model, the next 20% choose the threshold, the latest 20% are scored once and produce every number in the README and `training/metrics.json`. Home cities still come from the training period only.

## Consequences
- Test precision 96.6%, recall 96.6%, PR-AUC 0.992, with the threshold (0.578) chosen without seeing those rows. Validation gave 96.1% and 97.3%, so the choice carries over.
- The model trains on 90,000 rows instead of 120,000. PR-AUC moved from 0.9925 to 0.9918.
- `metrics.json` records both the validation and test figures, so the gap is visible.
