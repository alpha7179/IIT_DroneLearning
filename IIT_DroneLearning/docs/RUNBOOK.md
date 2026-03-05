# RUNBOOK.md - Daily Ops (Pursuer RL)

## 1) Start Checklist
- Confirm branch is `work/pursuer`
- Read `AGENTS.md` and `PROJECTS.md`
- Select one capability for this work block

## 2) Training Templates
```bash
# Stage1
mlagents-learn python/config/pursuer_s1_ray.yaml \
  --run-id=pursuer_s1_<exp>_<seed> --force

# Stage2 (warm-start from Stage1)
mlagents-learn python/config/pursuer_s2_vision.yaml \
  --run-id=pursuer_s2_<exp>_<seed> \
  --initialize-from=pursuer_s1_<exp>_<seed> --force

# Stage3 (warm-start from Stage2)
mlagents-learn python/config/pursuer_s3_lstm.yaml \
  --run-id=pursuer_s3_<exp>_<seed> \
  --initialize-from=pursuer_s2_<exp>_<seed> --force

# Stage4 (warm-start from Stage3)
mlagents-learn python/config/pursuer_s4_domainrand.yaml \
  --run-id=pursuer_s4_<exp>_<seed> \
  --initialize-from=pursuer_s3_<exp>_<seed> --force
```

## 3) Eval Minimum
- Seeds: `10` fixed
- Episodes: `30`
Required metrics:
- `catch_rate`
- `mean_time_to_catch`
- `crash_rate`
- `occlusion_reacquisition_time` (Stage2+)

## 4) Artifact Policy
Do not commit runtime outputs:
- `python/results/`
- `python/models/`
- `python/runs/`
- `python/checkpoints/`
- tensorboard event files

## 5) End Checklist
- Save one eval snapshot
- Append one line to `docs/EXPERIMENTS.md`
- Append one line to `docs/HANDOFF.md`
- Update `docs/TASKS.md` if priorities changed
