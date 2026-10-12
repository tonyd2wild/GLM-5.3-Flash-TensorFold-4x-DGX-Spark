#!/usr/bin/env bash
# serve_exl3.sh -- the fleet endpoint as served: GLM-5.3-Flash EXL3 (brandonmusic/GLM-5.3-Flash-tr3-4bpw) on the
# multi-stream engine, Reddie rank 0 on :8000 as glm-5.3-flash.
#   TP=4 (default): all four Sparks, 8 streams.   TP=2: Reddie and Spark4, 4 streams.   500K context a request either way.
#   TF_GLM_DENSE=q4 (default): the checkpoint's BF16 non-expert matrices (attention, indexer, shared and dense MLP) as
#   4-bit groups of 64, the head in FP8, kv_b and the embedding BF16. DENSE=fp8 holds them as FP8 (a scale per 128 block).
# Weights path: the NFS mount every rank sees (Reddie exports /var/tmp/models).
set -euo pipefail
cd "$(dirname "$0")"
NAME=tfmia TP=${TP:-4} PORT=${PORT:-8000} MODEL=${MODEL:-/mnt/reddie-models/GLM-5.3-Flash-EXL3-tr3-4bpw} \
  EXTRA_ENV="TF_GLM_DENSE=${DENSE:-q4} ${EXTRA_ENV:-}" PROFILE=multi \
  DRAFTS=auto VISION=1 CONTEXT=${CONTEXT:-500000} EFFORT=${EFFORT:-low} KV=fp8 ./tf_tp4.sh "${1:-serve}"
