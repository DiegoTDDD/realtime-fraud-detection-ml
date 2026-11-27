# 0002. `distance_km_from_home` as the impossible-travel signal

**Status:** Accepted

## Context
high_amount and card_testing were easy to separate. impossible_travel was not: the event stream has no home-city field, and about 13.5% of legitimate transactions also happen away from the card's usual city.

## Decision
Infer each card's home city as the city it uses most in the training period, and use the great-circle distance from that city as the feature.

## Alternatives considered
Each option was measured on the same held-out period. This comparison predates [ADR 0007](0007-validation-split-for-the-threshold.md); a design choice like this one is now made on the validation period, so the test period stays untouched:

| Feature | impossible_travel recall | Why it was not kept |
|---|---|---|
| Binary `is_unusual_city` | 0.59 to 0.65 | Cannot tell a nearby trip from a cross-country one |
| `implied_speed_kmh` (distance from the previous transaction ÷ elapsed time) | 0.62 | The previous transaction is often hours earlier, so the speed looks plausible |
| `distance_km_from_home` | 0.65 | Kept |

## Consequences
- The best of the three, but still 64.7% recall (64.2% on the test period after the re-split of ADR 0007). More signal would need data the stream does not have, such as travel history or device location.
- The home-city map is saved to `training/card_home_city.json` and loaded unchanged by the scoring job.
