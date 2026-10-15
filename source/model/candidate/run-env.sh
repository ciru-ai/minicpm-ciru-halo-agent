#!/usr/bin/env bash
set -euo pipefail
export VLLM_SOURCE=/opt/ciru/glm53-iu4/vllm VLLM_VENV=/opt/ciru/glm53-iu4/venv AITER_SOURCE=/opt/ciru/glm53-iu4/aiter
source /opt/ciru/glm53-iu4/runtime-env.sh
unset VLLM_SOURCE VLLM_VENV
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/srv/ssd/p3700ba/scratch/crown/minicpm-dspark-20261012/candidate
parent=/srv/ssd/sn850x/scratch/crown/ornith-ultra-20261007
export PYTHONPATH="${CIRU_Q8_ADAPTER:-$root}:$root:/srv/ssd/p3700ba/scratch/crown/model-adoption-q8-20261012/deps:$parent/integration/upstream-engine-ed7ae705/plugin-site:$parent${PYTHONPATH:+:$PYTHONPATH}"
exec "$parent/integration/upstream-engine-ed7ae705/venv/bin/python" "$@"
