# 0005. Class weights instead of resampling; PR-AUC as the main metric

**Status:** Accepted

## Context
7.5% of rows are fraud. Accuracy is meaningless at that balance: predicting "never fraud" scores 92.5%.

## Decision
Train XGBoost with `scale_pos_weight` = negatives ÷ positives (12.17), and report PR-AUC plus precision and recall at the chosen threshold, broken down by pattern.

## Alternatives considered
- **SMOTE or undersampling.** Changes the data the model sees. Synthetic minority rows would also break the velocity features, which depend on real neighbouring transactions.
- **ROC-AUC.** Looks good when negatives dominate, because the false-positive rate stays small even with many false alarms. PR-AUC reflects alert quality directly.

## Consequences
- Training data keeps its real distribution, and probabilities rank well (PR-AUC 0.9925).
- The weighted model's probabilities are not calibrated. They rank transactions but are not frequencies, so the alert threshold has to be chosen explicitly ([0004](0004-threshold-policy.md)).
