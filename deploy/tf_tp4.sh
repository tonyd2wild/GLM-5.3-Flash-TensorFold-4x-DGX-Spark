#!/usr/bin/env bash
# tf_tp4.sh -- GLM-5.3-Flash on our TensorFold build, tensor-parallel 4 over the four Sparks.
# Run on Reddie (rank 0, HTTP on :8000). Followers start first; every rank reads the weights
# from the same path (local on Reddie, NFS from Reddie on the others).
#
#   tf_tp4.sh serve | stop | status | logs [rank]
#
# Env overrides: CONTEXT (500000), EFFORT (low), VISION (1), DRAFTS (mtp|none), EXTRA (more serve args)
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
EXTRA=${EXTRA:-}
EXTRA_ENV=${EXTRA_ENV:-}                         # more container env, e.g. "NCCL_PROTO=LL NCCL_MAX_NCHANNELS=8"
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
  local r=$1 a="/model --tp 4 --rank $r --master $MASTER --master-port $MASTER_PORT --context $CONTEXT"
  a+=" --reasoning-effort $EFFORT"
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
    for r in 3 2 1; do start_rank "$r"; done
    start_rank 0
    echo "rank 0 serves http://0.0.0.0:$PORT as $SERVED (context $CONTEXT, effort $EFFORT, vision $VISION, drafts $DRAFTS)"
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
