# Tasks: Honest threshold, state that survives restarts, and model monitoring

## Phase 1: Blocking
- [x] T001 [US4] Parse the wire format and plain JSON in the scoring job

## Phase 2: User Story 1 (P1)
- [x] T002 Three-way chronological split; threshold on validation; metrics on test; retrain and record
- [x] T003 [P] Tests for the split and that the threshold ignores the test period

## Phase 3: User Story 2 (P2)
- [x] T004 Velocity state function and applyInPandasWithState with timeout
- [x] T005 Delta predictions with txnAppId/txnVersion
- [x] T006 [P] Unit test of the state function; Spark restart test against batch features
- [x] T007 Dashboard and sample export read Delta

## Phase 4: User Story 3 (P3)
- [x] T008 Reference distribution saved at training; monitor.py; tests

## Phase 5: Polish
- [x] T009 Pipeline check: registry, restart, duplicates, monitor, Spark tests
- [x] T010 ADRs 0007-0009 (0003 superseded); README; model card
