#!/usr/bin/env bash
# tf_tp4.sh -- GLM-5.3-Flash on our TensorFold build, tensor-parallel 4 over the four Sparks (TP=2: Reddie and Spark4).
# Run on Reddie (rank 0, HTTP on :8000). Followers start first; every rank reads the weights
# from the same path (local on Reddie, NFS from Reddie on the others).
#
#   tf_tp4.sh serve | stop | status | logs [rank]
#
# Env overrides: TP (4|2), CONTEXT (500000), EFFORT (low), VISION (1), DRAFTS (mtp|none), KV (fp8|bf16), EXTRA (more serve args)
set -euo pipefail

SRC=${SRC:-/srv/glm53-tensorfold/src}            # our engine tree, the same path on every node
CACHE=${CACHE:-/srv/glm53-tensorfold/cache}      # torch extensions, Triton and kernel caches (survive restarts)
MODEL=${MODEL:-/mnt/reddie-models/glm-quant-mix-derisked/lossless8}
IMAGE=${IMAGE:-glm53-roce:v11-b58f34ea}          # torch 2.13 (CUDA 13.0), NCCL 2.29, nvcc 13.0
NAME=${NAME:-tfglm}
PORT=${PORT:-8000}
SERVED=${SERVED:-glm-5.3-flash}
CONTEXT=${CONTEXT:-500000}
EFFORT=${EFFORT:-low}
VISION=${VISION:-1}
DRAFTS=${DRAFTS:-auto}                           # none | mtp | auto (MTP and DFlash2, the engine picks each round)
DRAFTER=${DRAFTER:-/mnt/reddie-models/GLM-5.3-Flash-DFlash2}   # incoai's DFlash2, read over NFS like the weights
KV=${KV:-fp8}                                    # DSA cache: fp8 (e4m3 latent + pooled index keys) | bf16 (fallback)
EXTRA=${EXTRA:-}
EXTRA_ENV=${EXTRA_ENV:-}                         # more container env, e.g. "NCCL_PROTO=LL NCCL_MAX_NCHANNELS=8"
PARALLEL=${PARALLEL:-1}                          # concurrent requests (the multi-stream engine: 2-8)
# PROFILE=multi: the multi-stream engine's four-Spark settings (MiaAI-Lab's GLM patches, Apache-2.0, credited in
# CREDITS.md), on our checkpoint: 8 streams, FP8 KV (TF_GLM_KV), RoCE all-gathers on the switch, split prefill, copy drafts.
PROFILE=${PROFILE:-single}
TP=${TP:-4}                                      # 4: all four Sparks; 2: Reddie (rank 0) and Spark4 (rank 1)
if [[ $PROFILE == multi ]]; then
  SRC=${SRC_MULTI:-/srv/glm53-tensorfold/src-mia}      # the multi-stream engine's tree (see engine-multi/)
  CACHE=${CACHE_MULTI:-/srv/glm53-tensorfold/cache-mia}
  EXTRA="--max-tokens ${MAX_TOKENS:-32768} $EXTRA"
  # the settings that depend on the Spark count (MiaAI-Lab's defaults for each): streams, the prompt split's own front and
  # micro-batches, 4,096-row prompt chunks and the decode side stream at four Sparks; the kept-state pool and reserve
  if (( TP == 4 )); then
    PARALLEL=${PARALLEL_MULTI:-8}
    TP_ENV="TF_GLM_PROMPT_OWN_FRONT=1 TF_GLM_PROMPT_MICROBATCH=1 TF_GLM_PREFILL_ROWS=4096 TF_GLM_SIDE=1 TF_GLM_MULTI_WINDOW=64 \
TF_GLM_CACHE_GIB=32 TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE:-19.6}"
  else
    PARALLEL=${PARALLEL_MULTI:-4}
    TP_ENV="TF_GLM_CACHE_GIB=12.5 TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE:-14.5}"
  fi
  EXTRA_ENV="TF_GLM_MTP=auto TF_GLM_KV=$KV TF_GLM_COMM=roce TF_ROCE_HCA=rocep1s0f0 TF_ROCE_GID_INDEX=3 TF_ROCE_RING=0 \
TF_ROCE_MAX_KB=${ROCE_MAX_KB:-1024} TF_ROCE_WAIT_S=300 TF_GLM_COPY_DRAFTS=1 TF_GLM_COPY_MAX=15 TF_GLM_DFLASH_POLICY=fnc7:0.3 \
TF_GLM_HC_SPLIT=1 TF_GLM_PREFILL_OVERLAP=2 TF_GLM_PROMPT_PARTIALS=fp32 TF_GLM_MSA=0 TF_GLM_KDA_CHUNKED=1 \
TF_GLM_WIDE_GRAPHS=16 TF_GLM_COPY_REPLY_MATCH=16 TF_GLM_MULTI_LONE=0 TF_GLM_CACHE_ENTRIES=32 TF_GLM_MULTI_PREFILL=1 \
TF_GLM_STREAM_SMOOTH=1 TF_GLM_STREAM_SMOOTH_MS=400 TF_GLM_FILL_BUDGET_MS=200 TF_GLM_FILL_DRAFTS=1 TF_GLM_L2PF=1 \
TF_GLM_SHARED_PREFIX=1 $TP_ENV TENSORFOLD_NUCLEUS_UNION=1 TENSORFOLD_NO_UPDATE_CHECK=1 $EXTRA_ENV"
fi
# the image's CUDA toolkit lacks the cuSPARSE/cuSOLVER headers torch's CUDAContext includes; torch's pip bundle has them
CUDA_PIP_INC=/cache/cuda_extra_inc     # only the headers the toolkit lacks, copied from torch's pip bundle (nvidia/cu13/include)
MASTER=192.168.192.2
MASTER_PORT=${MASTER_PORT:-29551}
# rank -> ssh target (Reddie runs locally); the order matches the live GLM's ranks
HOSTS=(local tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1)

run() {  # run <rank> <command>
  local h=${HOSTS[$1]}
  if [[ $h == local ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$h" "$2"; fi
}

serve_args() {  # serve_args <rank>
  local r=$1
  local a="/model --tp $TP --rank $r --master $MASTER --master-port $MASTER_PORT --context $CONTEXT"
  a+=" --reasoning-effort $EFFORT"
  [[ $PROFILE == single ]] && a+=" --kv-dtype $KV"         # the multi engine reads TF_GLM_KV instead
  (( PARALLEL > 1 )) && a+=" --parallel $PARALLEL"
  [[ $DRAFTS == none ]] && a+=" --no-drafts"
  [[ $DRAFTS == auto ]] && a+=" --drafter /drafter"
  [[ $VISION == 1 ]] && a+=" --vision"
  if [[ $r == 0 ]]; then a+=" --host 0.0.0.0 --port $PORT --name $SERVED"; fi
  echo "$a $EXTRA"
}

start_rank() {
  local r=$1
  local model_real drafter_mount="" envs=""
  for kv in $EXTRA_ENV; do envs+=" -e $kv"; done
  model_real=$(run "$r" "readlink -f $MODEL")
  [[ $DRAFTS == auto ]] && drafter_mount="-v $(run "$r" "readlink -f $DRAFTER"):/drafter:ro"
  run "$r" "mkdir -p $CACHE/torch_ext $CACHE/triton $CACHE/home && docker rm -f $NAME-r$r >/dev/null 2>&1; \
    docker run -d --name $NAME-r$r --gpus all --network host --ipc host \
      --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 --cap-add IPC_LOCK \
      --device /dev/infiniband --memory 112g --memory-swap 112g \
      -e NCCL_CUMEM_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA==rocep1s0f0 \
      -e NCCL_IB_GID_INDEX=3 -e NCCL_NET=IB -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 \
      -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
      -e NCCL_NET_PLUGIN=none -e NCCL_DEBUG=WARN $envs \
      -e MAX_JOBS=2 -e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e TRITON_CACHE_DIR=/cache/triton -e HOME=/cache/home \
      -e NVCC_APPEND_FLAGS=-I$CUDA_PIP_INC -e CPATH=$CUDA_PIP_INC \
      -e PYTHONPATH=/tf/src -e PYTHONUNBUFFERED=1 \
      -v $SRC:/tf:ro -v $CACHE:/cache -v $model_real:/model:ro $drafter_mount \
      --entrypoint python3 $IMAGE -m tensorfold.cli serve $(serve_args "$r")" >/dev/null
  echo "rank $r started on ${HOSTS[$r]}"
}

case "${1:-}" in
  serve)
    # stop every rank first (rank 0 first): a follower started beside the old rank 0 joins its rendezvous and dies
    for r in 0 1 2 3; do run "$r" "docker rm -f $NAME-r$r >/dev/null 2>&1 || true"; done
    sleep 3
    for (( r = TP - 1; r >= 1; r-- )); do start_rank "$r"; done
    start_rank 0
    echo "rank 0 serves http://0.0.0.0:$PORT as $SERVED (tp $TP, context $CONTEXT, effort $EFFORT, vision $VISION, drafts $DRAFTS, kv $KV, profile $PROFILE, parallel $PARALLEL)"
    ;;
  stop)
    for r in 0 1 2 3; do run "$r" "docker rm -f $NAME-r$r >/dev/null 2>&1 || true"; done
    ;;
  status)
    for r in 0 1 2 3; do echo "rank $r: $(run "$r" "docker ps -a --filter name=^$NAME-r$r\$ --format '{{.Status}}'")"; done
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" || echo "(no HTTP yet)"
    ;;
  logs)
    r=${2:-0}; run "$r" "docker logs --tail 80 $NAME-r$r 2>&1"
    ;;
  *) echo "usage: $0 serve|stop|status|logs [rank]"; exit 2 ;;
esac
