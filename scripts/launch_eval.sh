#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment/main.yaml}"
CKPT="${2:-runs/main/checkpoint.pt}"

python -m trustport.wheelhouse evaluate --config "${CONFIG}" --checkpoint "${CKPT}"
