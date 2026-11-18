# 0001. Generate training data in batch with the producer's own logic

**Status:** Accepted

## Context
The live pipeline's Bronze layer held only 189 fraud examples, too few for stable metrics. Running the producer through Kafka and Spark for hours to collect more would be slow and would not be reproducible.

## Decision
`training/generate_training_data.py` reuses the producer's simulation logic (same cards, cities, categories and fraud patterns) and writes 150,000 transactions straight to Parquet in about three seconds. The seed and the start date are fixed, so every run writes the same data.

## Alternatives considered
- **Collect from the live stream.** Realistic plumbing, but it takes hours and gives a different dataset every time.
- **Public fraud datasets (e.g. the Kaggle credit-card set).** Real data, but its features are anonymised PCA components that the live stream does not produce, so the model could not score the stream.

## Consequences
- Training data and the live stream come from the same distribution, so offline and live metrics can be compared.
- The same simplicity makes the task easier than real fraud (see the model card's limitations).
- If the producer's logic changes, the generator has to be updated by hand.
