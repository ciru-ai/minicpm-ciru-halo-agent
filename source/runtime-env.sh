#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VLLM_SOURCE=/opt/ciru/glm53-iu4/vllm VLLM_VENV=/opt/ciru/glm53-iu4/venv AITER_SOURCE=/opt/ciru/glm53-iu4/aiter
source /opt/ciru/glm53-iu4/runtime-env.sh
unset VLLM_SOURCE VLLM_VENV
mini="$root/payload/model"
parent="$root/payload/ornith"
engine="$parent/integration/upstream-engine-ed7ae705"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${CIRU_Q8_ADAPTER:-$mini/candidate}:$mini/candidate:$root/payload/deps:$engine/plugin-site:$parent${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${CIRU_PACKAGE_AUDIT:-0} == 1 ]]; then
    export CIRU_PACKAGE_ROOT="$root"
    exec "$engine/venv/bin/python" "$root/audit_entry.py" "$@"
fi
exec "$engine/venv/bin/python" "$@"
