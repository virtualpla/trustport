#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment/main.yaml}"
OUT="${2:-runs/main}"
STEPS="${3:-200}"

python -m trustport.wheelhouse train --config "${CONFIG}" --out "${OUT}" --steps "${STEPS}"
