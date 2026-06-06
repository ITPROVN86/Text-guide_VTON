#!/usr/bin/env bash
python -m src.infer \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/st_vton_best.pth \
  --split test \
  --save_dir outputs/inference
