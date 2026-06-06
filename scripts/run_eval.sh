#!/usr/bin/env bash
python -m src.eval \
  --config configs/default.yaml \
  --pred_dir outputs/inference \
  --split test \
  --save_csv outputs/metrics.csv
