# AGENTS.md - Pursuer RL Contract (Semester Frozen)

This file is the semester contract for Pursuer/Tracker work.
Do not modify during the semester. Record change requests in `docs/DECISIONS.md` or RFC notes.

## 1) Mandatory Read Order
1. `AGENTS.md`
2. `PROJECTS.md`
3. `docs/RUNBOOK.md`
4. `docs/TASKS.md`
5. `docs/EXPERIMENTS.md`

## 2) Ownership
In scope:
- Pursuer observation/action/reward/termination
- Occlusion handling from pursuer side
- Stage training execution on Colab Pro+
- Evaluation metric tracking

Out of scope:
- World map or level design
- Sensor rendering internals
- Evader ownership
- Large structure refactors without agreement

## 3) Version and Branch Lock
- Branch: `work/pursuer`
- Unity: `6.0.57f1`
- ML-Agents: `4.0.x`
- Python: `3.10`
- PyTorch: `2.x`
- Physics loop: `FixedUpdate 50Hz (dt=0.02)`

## 4) Sim2Real Control Rule
Policy must output setpoints, not raw motor signals.
Required path:
`policy -> PID (rate/attitude) -> Rigidbody force/torque`

Keep action scaling, clipping, and timing identical across train/eval.

## 5) Interface Contract
### Action (Pursuer)
- Continuous 4D
- `thrust_cmd in [0,1]`
- `roll_rate_cmd in [-1,1]`
- `pitch_rate_cmd in [-1,1]`
- `yaw_rate_cmd in [-1,1]`

### Observation (Pursuer)
Always:
- Self: local pos/vel, angular vel, orientation, altitude
- Ray: distance + tag (`building`, `ground`, `drone`, `nofly`, `goal`)

Stage1 only:
- Relative target pos/vel ground truth (remove in Stage2+)

Stage2+:
- `is_visible`
- `last_seen_rel_pos`
- `last_seen_rel_vel`
- `time_since_seen`
- RGB `84x84` (optional stack 2-4)

### Termination
- Catch: distance `< 1.5m` for `5` consecutive steps
- Crash: altitude `< 0.5m`, tilt `> 70deg`, or hard collision
- Timeout: `25s`
- End on out-of-bounds/no-fly

## 6) Reward Contract
Stage1-2 baseline:
- `+1.0` catch
- `-1.0` crash
- `-0.2` timeout
- `-0.001` step
- `+0.02 * (d_prev - d_now)` shaping

Stage2+:
- `+0.2` reacquisition bonus on `is_visible: 0 -> 1`
- Add speed/angular penalties only if instability is proven

## 7) Stage Order and Gates
Order is fixed:
`Stage1 Ray -> Stage2 Vision -> Stage3 LSTM -> Stage4 Domain Randomization`

Gate expectations:
- Stage1: stable pursuit and collision handling
- Stage2: GT hint removed, vision warm-start works
- Stage3: improved occlusion reacquisition time
- Stage4: acceptable drop on unseen maps/seeds

## 8) Training and Eval Standard
Folder layout:
```text
python/
  config/
  scripts/
  results/
  models/
```

Naming:
- Config: `pursuer_s{stage}_{yyyymmdd}_{short}.yaml`
- Run ID: `pursuer_s{stage}_{exp}_{seed}`

Minimum eval:
- 10 fixed seeds
- 30 episodes
- Metrics: `catch_rate`, `mean_time_to_catch`, `crash_rate`, `occlusion_reacquisition_time`

Log every run in `docs/EXPERIMENTS.md`:
`date | git_commit | stage | run_id | seed | catch_rate | mean_ttc | crash_rate | reacq_time | note`

## 9) Dependency Contract
World:
- Deterministic reset
- Stable physics timing
- Catch/collision API
- PID to force/torque path

Sensor:
- Stable ray tag schema
- LoS visibility API
- Headless camera support

Evader:
- Scripted baseline for early stages
- Stable episode sync for RL-vs-RL

## 10) Debug Order
1. Early termination spikes
2. Catch condition validity
3. Action scale mismatch
4. Reward sign/magnitude errors
5. Observation NaN/Inf or frame mismatch
6. Decision period vs timestep mismatch
7. Train/eval drift

## 11) Semester Deliverables
- Final Pursuer ONNX and stage configs
- Stage metric table (catch/time/crash/occlusion)
- Reproducible Colab scripts
- Occlusion demo evidence
