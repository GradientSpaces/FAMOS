#!/usr/bin/env bash
# Usage: bash scripts/train/train_stage2.sh [Hydra overrides...]
# Run from the repository root.
#
# Stage 2: 80,000 optimizer steps on an equal procedural/real mixture
# (config/train_stage2.yaml), initialized from FAMOS_STAGE1_CHECKPOINT.
# Writes to OUTPUT_DIR and resumes from OUTPUT_DIR/last.ckpt if it exists.
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/famos_stage2}"
python train.py --config-name train_stage2 "log_dir=${OUTPUT_DIR}" "$@"
