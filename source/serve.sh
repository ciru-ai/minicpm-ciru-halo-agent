#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mini="$root/payload/model"
release="$mini/20261014-haloloom-native-adapter"
python3 "$root/VERIFY.py"
export CIRU_DSPARK_RUNNER=1 CIRU_DSPARK_WINDOW=1024 CIRU_DSPARK_KEEP_PREFIX_TAIL=1 CIRU_DSPARK_SAMPLE=greedy CIRU_DSPARK_ATTENTION=retained
export CIRU_GDN_MODE=clean CIRU_GDN_PROFILE=0 CIRU_BF16_PREFILL_MODE=clean CIRU_BF16_PROFILE=0
export CIRU_ATTENTION_CHILD=shared16 CIRU_ATTENTION_CHILD_AUDIT=0 CIRU_DSPARK_KV_GIB=24
export CIRU_TOKENIZE_FIX=1 CIRU_SERVING_MODE=candidate CIRU_SERVING_MANIFEST="$mini/20261013-haloloom-gdn-remainder/specializations.json"
export CIRU_SERVING_RUN="$root/state/serving" CIRU_DEPTH_ADAPTIVE=1 CIRU_DEPTH_WORKER=model60_attention_warmup_worker.Worker CIRU_REMAINDER_MODE=clean
export CIRU_METADATA_PARTITION_AUDIT=0
mkdir -p "$CIRU_SERVING_RUN"
unset CIRU_DSPARK_TRACE CIRU_DSPARK_KV_AUDIT CIRU_DSPARK_ADAPTIVE CIRU_DEPTH_CONTROL CIRU_DEPTH_RUN CIRU_NATIVE_PROFILE CIRU_ORDERED_ADMISSION CIRU_ATTENTION_W8_AUDIT CIRU_CONV_GRID_AUDIT
export PYTHONPATH="$release/attention-warmup-release:$release/metadata-partition-release:$release/small-prefill-release:$release/workflow-release:$release/draft-head-release:$release/conv-release:$release/attention-release:$release/routed-release"
exec python3 "$mini/20261013-haloloom-gdn-remainder/launch.py" minicpm --drafts 3 --attention retained --prefill retained --variant installed-rc "$@"
