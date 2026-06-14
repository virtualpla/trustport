#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data}"
mkdir -p "${ROOT}"

echo "Place the credentialed corpora under ${ROOT} following docs/project-context.md section 6:"
echo "  ${ROOT}/mimiciv     (PhysioNet credentialed)"
echo "  ${ROOT}/n2c2_2022    (DUA required)"
echo "  ${ROOT}/bc5cdr ${ROOT}/chemprot ${ROOT}/ccks ${ROOT}/medbench"
echo "Synthetic cohorts are generated on demand and need no download."
