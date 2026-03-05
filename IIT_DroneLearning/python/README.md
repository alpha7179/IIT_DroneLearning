# Python Pipeline (Pursuer RL)

## Roles
- `python/config/`: stage YAML configs
- `python/scripts/`: train/eval/export scripts
- `python/results/`: runtime logs (ignored)
- `python/models/`: checkpoints and ONNX (ignored)

## Naming
- Config: `pursuer_s{stage}_{yyyymmdd}_{short}.yaml`
- Run ID: `pursuer_s{stage}_{exp}_{seed}`

## Minimum Files
- `python/config/pursuer_s1_ray.yaml`
- `python/config/pursuer_s2_vision.yaml`
- `python/config/pursuer_s3_lstm.yaml`
- `python/config/pursuer_s4_domainrand.yaml`
- `python/scripts/train_stage.sh`
- `python/scripts/eval_pursuer.py`
