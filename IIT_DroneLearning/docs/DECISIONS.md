# DECISIONS.md - Frozen Technical Decisions

## D-001 Branch Policy
- Decision: Use `work/pursuer` only for Pursuer stream
- Reason: Reduce branch drift and review overhead

## D-002 Stage Order
- Decision: `Stage1 Ray -> Stage2 Vision -> Stage3 LSTM -> Stage4 Domain Randomization`
- Reason: Stabilize before adding high-variance components

## D-003 Sim2Real Control
- Decision: Setpoint policy output, PID to force/torque mapping
- Reason: Better transferability and safer constraints

## D-004 Eval Baseline
- Decision: 10 fixed seeds and 30 episodes minimum
- Reason: Comparable metrics across runs
