#!/usr/bin/env bash
# serve_exl3.sh -- the fleet endpoint as served: GLM-5.3-Flash EXL3 (brandonmusic/GLM-5.3-Flash-tr3-4bpw) on the
# multi-stream engine, TP4 over the four Sparks, Reddie rank 0 on :8000 as glm-5.3-flash.
#   TF_GLM_DENSE=fp8: the checkpoint's BF16 non-expert matrices (attention, indexer, shared and dense MLP, head) held as
#   FP8 with a scale per 128 block; kv_b and the embedding stay BF16. q4 (MiaAI-Lab's default) trades quality for speed.
# Weights path: the NFS mount every rank sees (Reddie exports /var/tmp/models).
set -euo pipefail
cd "$(dirname "$0")"
NAME=tfmia PORT=${PORT:-8000} MODEL=${MODEL:-/mnt/reddie-models/GLM-5.3-Flash-EXL3-tr3-4bpw} \
  EXTRA_ENV="TF_GLM_DENSE=${DENSE:-fp8} ${EXTRA_ENV:-}" PROFILE=multi PARALLEL_MULTI=${PARALLEL_MULTI:-8} \
  DRAFTS=auto VISION=1 CONTEXT=${CONTEXT:-500000} EFFORT=${EFFORT:-low} KV=fp8 ./tf_tp4.sh "${1:-serve}"
