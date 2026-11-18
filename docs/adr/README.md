# Architecture decision records

| # | Decision | Status |
|---|---|---|
| [0001](0001-synthetic-training-data.md) | Generate training data in batch with the producer's own logic | Accepted |
| [0002](0002-distance-feature.md) | `distance_km_from_home` as the impossible-travel signal | Accepted |
| [0003](0003-process-local-velocity-state.md) | Keep per-card velocity state in the scoring process | Accepted |
| [0004](0004-threshold-policy.md) | Choose the threshold by mean recall across patterns under a 95% precision floor | Accepted |
| [0005](0005-class-weights-and-pr-auc.md) | Class weights instead of resampling; PR-AUC as the main metric | Accepted |
| [0006](0006-shared-features-and-time-split.md) | One feature module for training and serving; time-based split with home cities from training rows only | Accepted |
