#!/usr/bin/env bash
# Usage: bash scripts/train/train_stage1.sh [Hydra overrides...]
# Run from the repository root.
#
# Stage 1: 50,000 optimizer steps on procedural objects (config/train_stage1.yaml).
# Writes to OUTPUT_DIR and resumes from OUTPUT_DIR/last.ckpt if it exists.
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/famos_stage1}"
python train.py --config-name train_stage1 "log_dir=${OUTPUT_DIR}" "$@"
