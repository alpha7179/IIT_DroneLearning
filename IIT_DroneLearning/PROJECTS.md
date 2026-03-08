# PROJECTS.md - Pursuer RL Execution Board

Primary planning board for branch `work/pursuer`.
Use with `AGENTS.md` for all design and implementation decisions.

## 1) Working Docs
- `docs/README.md`
- `docs/RUNBOOK.md`
- `docs/REQUEST_TEMPLATE.md`
- `docs/TASKS.md`
- `docs/EXPERIMENTS.md`
- `docs/HANDOFF.md`
- `docs/DECISIONS.md`

## 2) Project Objective
- Train a robust Pursuer policy for urban occlusion scenarios.
- Preserve Sim2Real transfer through hierarchical control and fixed interfaces.

## 3) Role Split
- `World`: map, spawn/reset, termination infrastructure
- `Sensor`: ray/vision/LoS and noise pipeline
- `Evader`: escape policy and baseline
- `Pursuer`: tracking, interception, occlusion recovery

## 4) Stage Plan and Exit Gates
Stage order:
`Stage1 Ray -> Stage2 Vision -> Stage3 LSTM -> Stage4 Domain Randomization`

Stage1 exit:
- catch rate `>= 0.70` on fixed seeds
- crash rate `<= 0.10`

Stage2 exit:
- catch rate `>= 0.50` on obstacle maps
- stable short-occlusion recovery

Stage3 exit:
- lower `occlusion_reacquisition_time`
- stable results across at least 3 seeds

Stage4 exit:
- controlled degradation on unseen maps/seeds

## 5) Runtime Baseline
- Unity `6.0.57f1`
- ML-Agents `4.0.x`
- Python `3.10`

```bash
pip install -U pip
pip install mlagents==4.0.* torch tensorboard
mlagents-learn python/config/pursuer_s1_ray.yaml --run-id=pursuer_s1_base_00 --force
mlagents-learn python/config/pursuer_s2_vision.yaml --run-id=pursuer_s2_base_00 --initialize-from=pursuer_s1_base_00 --force
```

## 6) Metrics and Reporting
Required metrics:
- `catch_rate`
- `mean_time_to_catch`
- `crash_rate`
- `occlusion_reacquisition_time` (Stage2+)

Run log:
- File: `docs/EXPERIMENTS.md`
- Format: `date | commit | stage | run_id | seed | metrics | note`

## 7) Risks
- Physics timing mismatch between train and eval
- Reward shaping dominating terminal signals
- Overfitting to known seeds/maps
- Interface drift between role owners

## 8) Done at Semester End
- ONNX export and reproducible config set
- Stage-wise evaluation summary
- Re-runnable Colab workflow
- Occlusion-focused demo output
