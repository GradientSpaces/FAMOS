#!/usr/bin/env bash
# Usage: bash scripts/eval/eval_partnet_mobility.sh [NUM_STATES] [Hydra overrides...]
# Run from the repository root. The checkpoint is read from FAMOS_CHECKPOINT.
set -euo pipefail

NUM_STATES=2
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    NUM_STATES="$1"
    shift
fi

python evaluate.py --config-name inference \
    data=partnet_mobility \
    data.max_states="$NUM_STATES" \
    log_dir="results/states_$NUM_STATES" \
    "$@"
