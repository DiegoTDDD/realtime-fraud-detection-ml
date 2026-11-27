# Feature Specification: Honest threshold, state that survives restarts, and model monitoring

**Feature Branch**: `001-production-scoring`

**Created**: 2026-11-27

**Status**: Implemented

**Input**: User description: "Bring the fraud scoring layer to senior level. The README's own
limits: the threshold was chosen on the test set, velocity state lives in one process's memory
and is lost on restart, and nothing acts on live precision and recall. Also, the topic now
carries the schema registry's wire format."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reported metrics the threshold never saw (Priority: P1)

The fraud team reads precision and recall in the README and trusts them to predict alert
volume. The threshold is chosen on a validation period; the test period is scored once.

**Independent Test**: The training split has three consecutive time periods; the threshold
comes from the middle one; metrics.json reports the last one.

**Acceptance Scenarios**:

1. **Given** the generated data, **When** the model is trained, **Then** the earliest 60% train,
   the next 20% choose the threshold, and the latest 20% produce the reported metrics.
2. **Given** home cities, **When** they are inferred, **Then** they come from the training
   period only.

---

### User Story 2 - A restart loses no history and writes nothing twice (Priority: P2)

The scoring job is redeployed or crashes. Card-testing bursts that straddle the restart are
still counted, and no transaction gets two predictions.

**Independent Test**: Stream events through the scoring transformation in two runs with the
same checkpoint; the velocity features equal those the training code computes on all events
at once, and every transaction has one prediction.

**Acceptance Scenarios**:

1. **Given** per-card velocity state in Spark's state store (applyInPandasWithState), **When**
   the job restarts from its checkpoint, **Then** the state is restored.
2. **Given** predictions written to Delta with the batch id as transaction version, **When** a
   batch is replayed after a crash, **Then** it is not appended again.
3. **Given** several workers, **When** cards are spread across them, **Then** each card's state
   lives with its key, not in one Python process.

---

### User Story 3 - Someone is told when the model degrades (Priority: P3)

**Acceptance Scenarios**:

1. **Given** the latest N predictions with their labels, **When** the monitor runs, **Then** it
   checks precision >= 0.90, recall >= 0.90 and every pattern's recall against its floor.
2. **Given** the input distribution saved at training, **When** live amounts, distances or
   velocities shift (PSI >= 0.25), **Then** the monitor fails with the feature named.
3. **Given** `ALERT_WEBHOOK_URL`, **When** a check fails, **Then** the failures are posted.

---

### User Story 4 - Read the registry wire format (Priority: P1, blocking)

**Acceptance Scenarios**:

1. **Given** messages with a 0 byte and a 4-byte schema id before the JSON, **When** the job
   parses them, **Then** it reads the JSON after the header.
2. **Given** a message without the header, **When** it is parsed, **Then** it is read as JSON
   as before.

### Edge Cases

- A card's first transaction after a restart sees its history from before the restart.
- A pattern with no examples in the monitoring window is skipped, not failed.
- Fewer than the minimum predictions in the window: the monitor says so and passes.

## Requirements *(mandatory)*

- **FR-001**: Chronological 60/20/20 split; threshold on validation; metrics on test.
- **FR-002**: Velocity features computed with applyInPandasWithState keyed by card, with a
  processing-time timeout that drops idle cards.
- **FR-003**: Predictions written to a Delta table idempotently per batch.
- **FR-004**: A Spark test runs the stateful transformation across two runs with one checkpoint
  and compares with the batch features.
- **FR-005**: `scoring/monitor.py` with written thresholds, PSI against a reference saved at
  training, exit code and webhook.
- **FR-006**: The parser accepts both the wire format and plain JSON.
- **FR-007**: The pipeline check registers the schema, kills and restarts the scoring job, and
  requires one prediction per transaction and a passing monitor.

## Success Criteria *(mandatory)*

- **SC-001**: README results are test-period numbers with a validation-chosen threshold.
- **SC-002**: The Spark test shows identical velocity features across a restart.
- **SC-003**: The pipeline check shows zero duplicate predictions after a SIGKILL restart.
- **SC-004**: The monitor passes on the live stream in CI and its rules are unit-tested.

## Clarifications

### Session 2026-11-27

- Q: Spark state or Redis? → A: Spark's state store: no extra service, checkpointed with the
  offsets, so state and input position always match.
- Q: Monitoring window? → A: the latest 2,000 predictions, at least 500 to judge.
- Q: Drift metric? → A: PSI over the training deciles, 0.25 as the usual "significant shift".
