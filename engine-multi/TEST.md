# mia-derisk: cluster test plan (TP4, our ModelOpt pack)

Branch `mia-derisk` = TensorFold v0.6.0 + MiaAI-Lab's 106 GLM patches (Apache-2.0) + our ModelOpt reader: NVFP4 routed
experts and MXFP8 / FP8-block projections run as stored in every forward (serial engine, `--parallel` verify windows,
multi-stream prompt chunks, sliced fills, the row split with overlap / own-row fronts / micro-batches, MTP). Checked
locally: `compileall`, `tests/test_glm_modelopt_cpu.py` (12 cases), the MoE / projection dispatch with torch
reference kernels. Nothing has run on a GPU yet.

## 0. Put the tree on every node (same path), its own kernel cache

```bash
git -C <this worktree> archive --format=tar.gz -o /tmp/mia-derisk.tgz mia-derisk       # on the Mac
# copy it to Reddie, then on Reddie:
sudo mkdir -p /srv/glm53-tf-mia && sudo chown $USER /srv/glm53-tf-mia && tar -xzf /tmp/mia-derisk.tgz -C /srv/glm53-tf-mia
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  rsync -a --mkpath /srv/glm53-tf-mia/ $h:/srv/glm53-tf-mia/; done      # src/tensorfold under it: tf_tp4.sh mounts SRC
```

Stop production yourself first (`deploy/tf_tp4.sh stop`): the test needs the GPUs. Restore it after (step 5).

## 1. Launch (our `deploy/tf_tp4.sh`, always `PROFILE=multi`)

This tree's CLI takes `--kv-dtype bf16|int8|int4` only; GLM's FP8 cache is `TF_GLM_KV=fp8`, which the multi profile
passes as env (so one stream is `PROFILE=multi PARALLEL_MULTI=1`). Three of the profile's values are off for this tree:
`TF_GLM_WIDE_GRAPHS=1` / `TF_GLM_COPY_REPLY_MATCH=1` are rejected at startup (Mia's COPY_CODE=1 maps to 16 and 16),
and with a 64-row window Mia raises `TF_ROCE_MAX_KB` to 1024 and the reserve to 19.6 GiB at 8 requests. Make a copy
rather than rely on duplicate `-e` names:

```bash
cd ~/GLM-5.3-Flash-TensorFold-DGX-Spark
sed -e 's/TF_GLM_WIDE_GRAPHS=1 TF_GLM_COPY_REPLY_MATCH=1/TF_GLM_WIDE_GRAPHS=16 TF_GLM_COPY_REPLY_MATCH=16/' \
    -e 's/TF_ROCE_MAX_KB=512/TF_ROCE_MAX_KB=1024/' \
    -e 's/TENSORFOLD_MEMORY_RESERVE_GIB=14.5/TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE:-14.5}/' \
    -e 's/TF_GLM_MTP=auto/TF_GLM_MTP=${MTP:-auto}/' \
    deploy/tf_tp4.sh > deploy/tf_tp4_mia.sh && chmod +x deploy/tf_tp4_mia.sh
COMMON="SRC=/srv/glm53-tf-mia CACHE=/srv/glm53-tf-mia-cache NAME=tfmia PORT=8011 SERVED=glm-5.3-flash-mia \
  PROFILE=multi KV=fp8 CONTEXT=500000 VISION=1 EFFORT=low"
env $COMMON PARALLEL_MULTI=1 DRAFTS=none deploy/tf_tp4_mia.sh serve              # A  serial reference, no drafts
env $COMMON PARALLEL_MULTI=1 DRAFTS=auto deploy/tf_tp4_mia.sh serve              # B  one stream, DFlash2 drafts
env $COMMON PARALLEL_MULTI=1 DRAFTS=auto MTP=1 deploy/tf_tp4_mia.sh serve      # B2 MTP and DFlash2
env $COMMON PARALLEL_MULTI=8 DRAFTS=auto RESERVE=19.6 deploy/tf_tp4_mia.sh serve # C  8 requests at once, DFlash2
```

Each rank runs `python3 -m tensorfold.cli serve /model --tp 4 --rank R --master 192.168.192.2 --master-port 29551
--context 500000 --reasoning-effort low [--parallel 8] [--no-drafts | --drafter /drafter] --vision`, rank 0 with
`--host 0.0.0.0 --port 8011 --name glm-5.3-flash-mia` (checked by a dry run of the copy), and the env below. Logs:
`NAME=tfmia deploy/tf_tp4_mia.sh logs R`.

### Mia's settings -> what the engine reads -> our launcher (TP=4, switch)

| Mia (`scripts/config.sh`, `start-tp4.sh`) | engine reads | our multi profile |
|---|---|---|
| TP=4, COMM=roce, FABRIC_EXPECT=mesh | `--tp 4`, TF_GLM_COMM=roce | same |
| PARALLEL (8 with DRAFTER=dflash2, 1 with mtp) | `--parallel N` | PARALLEL_MULTI |
| DRAFTER=dflash2 / mtp | `--drafter <DFlash2>` / none | DRAFTS=auto / mtp; none -> `--no-drafts` |
| TF_GLM_MTP=auto | TF_GLM_MTP | auto |
| KV=fp8 | TF_GLM_KV | fp8 (not `--kv-dtype`) |
| DENSE=q4 (TF_GLM_KVB unset) | TF_GLM_DENSE, TF_GLM_KVB | unset: EXL3 only, a ModelOpt pack ignores both |
| CONTEXT (1048576 with fp8 KV), MAX_TOKENS=32768 | `--context`, `--max-tokens` | 500000; not passed (4096) |
| THINKING=1, VISION=1, VISION_URLS=0 | `--thinking`, `--vision`, `--vision-urls` | default on, VISION=1 |
| nodes.sh per node | TF_ROCE_HCA, NCCL_IB_HCA, NCCL_SOCKET_IFNAME | rocep1s0f0, TF_ROCE_GID_INDEX=3, enp1s0f0np0 |
| TF_ROCE_MAX_KB 512, 16 x window past 32 rows | TF_ROCE_MAX_KB | 512 -> **1024** (window 64) |
| TF_ROCE_WAIT_S=300, TF_ROCE_RING=0 (switch) | same | same |
| COPY=1, COPY_MAX=15, DRAFT_POLICY=fnc7:0.3 | TF_GLM_COPY_DRAFTS, TF_GLM_COPY_MAX, TF_GLM_DFLASH_POLICY | same |
| SPLIT=1 | TF_GLM_HC_SPLIT=1, TF_GLM_PREFILL_OVERLAP=2 | same |
| (SPLIT=1, TP=4) | TF_GLM_PROMPT_OWN_FRONT=1, TF_GLM_PROMPT_MICROBATCH=1, TF_GLM_PREFILL_ROWS=4096, TF_GLM_PROMPT_PARTIALS=fp32 | same |
| TF_GLM_SIDE=1 (TP=4) | TF_GLM_SIDE | 1 (ModelOpt's FP8 shared experts stay on the main stream) |
| TF_GLM_MSA=0, KDA_CHUNKED=1 | TF_GLM_MSA, TF_GLM_KDA_CHUNKED | same |
| COPY_CODE=1 | TF_GLM_WIDE_GRAPHS=16, TF_GLM_COPY_REPLY_MATCH=16 | 1 and 1: **rejected**, use 16 |
| TF_GLM_MULTI_LONE=0, TF_GLM_MULTI_WINDOW (64 past 4 requests) | same | 0, 64 |
| TF_GLM_CACHE_ENTRIES=32, MULTI_PREFILL=1, SHARED_PREFIX=1 | TF_GLM_CACHE_ENTRIES, TF_GLM_MULTI_PREFILL, TF_GLM_SHARED_PREFIX | same |
| STREAM_SMOOTH=1 / _MS=400, FILL_BUDGET_MS=200, FILL_DRAFTS=1 | TF_GLM_STREAM_SMOOTH(_MS), TF_GLM_FILL_BUDGET_MS, TF_GLM_FILL_DRAFTS | same |
| TF_GLM_L2PF=1 | same | 1 |
| TF_GLM_EXL3_LOADS=nc, TF_GLM_EXL3_DEC_ORDER=0 | EXL3 only | unset |
| KV_POOL_GIB=32 (TP 3/4) | TF_GLM_CACHE_GIB | 32 |
| MEMORY_RESERVE_GIB = 14.5 + 0.95/request past 4 + 0.04/row past 32 | TENSORFOLD_MEMORY_RESERVE_GIB | 14.5 -> **19.6** at 8 requests, window 64 |
| NUCLEUS_UNION=1, DISPLAY_KV_MIB=0, SPILL_GIB=0 | TENSORFOLD_NUCLEUS_UNION, TF_GLM_DISPLAY_KV_MIB, `--spill-gib` | 1, unset, unset |

## 2. Expected log lines (rank 0)

- `loading glm-5.3-flash-mia: GLM-5.3-Flash (glm5_next) on CUDA, rank 0 of 4`; ranks 1-3 `rank R ready in ...s`
- `all-gathers up to 1024 KiB over RoCE (rank 1 over rocep1s0f0; ...)`
- **`ModelOpt checkpoint: NVFP4 routed experts in 42 layers, projections as stored (A BF16, B FP8-block, C MXFP8),
  kv_b, embed and head BF16; TF_GLM_DENSE / TF_GLM_KVB apply to EXL3 checkpoints only`**: C = 2 x KDA layers,
  B = 3 x DSA layers + 84 (shared experts). B or C at 0 means the 8-bit path was not taken. B2: 43 layers, MTP included.
- `TF_GLM_KV=fp8: the DSA latent cache and the indexer's pooled keys as e4m3 rows ...`
- `rows split between the ranks, exchanges by send/receive on a second stream ..., the next block's front on own rows
  meanwhile; DSA's replicated front ... on own rows ...; chunks of 4096+ rows in two halves ...`
- `verify windows up to 16 rows, CUDA graphs for 1, 2, 3, 4, 5, 6, 7, 8, 16 rows ...; copies from the reply match 16 tokens`
- `L2 prefetch in decode windows (bulk, up to 8 MiB a site): a .. sites, f .. sites, o .. sites`
- B, C: `drafter timings (ms, fastest of 7)` and `drafter costs (ms): verify ...`
- `vision: image and video input, a 1.0x GiB tower with ... GiB of workspace; ...`
- C: `--parallel 8: up to 8 requests decode together, their caches in one pool of N tokens (each up to ...)`
- `serving glm-5.3-flash-mia at http://0.0.0.0:8011/v1 on CUDA, rank 0 of 4`

Must not appear: `EXL3 support is experimental`, `split rule is ambiguous` (marker not dropped), `duplicate checkpoint
tensor` (stale shard copy counted), an `IndexError` in split.py (scalar scale), `TF_GLM_WIDE_GRAPHS: widths from 9`,
`the ranks were started with different settings`, any traceback.

## 3. Checks (run against each of A, B, C)

```bash
cat > /tmp/glm_check.py <<'EOF'
import base64, json, struct, sys, threading, time, urllib.request, zlib
url, model, out, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]          # mode: serial | concurrent
def png(w=224, h=112):                                     # left half red, right half blue
    row = b"\x00" + bytes((255, 0, 0) * (w // 2) + (0, 0, 255) * (w // 2))
    c = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + c(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + c(b"IDAT", zlib.compress(row * h)) + c(b"IEND", b""))
img = "data:image/png;base64," + base64.b64encode(png()).decode()
hay = " ".join(f"Line {i}: the harbour was grey and the boats were still." for i in range(1500))
needle = hay[:len(hay) // 2] + " The secret code is 4417. " + hay[len(hay) // 2:]
P = {"mul": "What is 17 * 23? Reply with the number only.",
     "primes": "List the first 10 prime numbers, comma separated, nothing else.",
     "code": "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Code only.",
     "count": "Count from 1 to 60, separated by spaces, nothing else.",
     "json": 'Give a JSON object with keys "city" and "country" for the capital of Japan. JSON only.',
     "needle": needle + "\nWhat is the secret code? Reply with the number only.",
     "vision": [{"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": "Name the colour of the left half, then the right half. Two words."}],
     "prose": "Write three sentences about a lighthouse at dawn."}
def ask(k):
    body = {"model": model, "messages": [{"role": "user", "content": P[k]}], "temperature": 0, "max_tokens": 1024,
            "reasoning_effort": "low"}
    r = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); m = json.load(urllib.request.urlopen(r, timeout=1800))["choices"][0]["message"]
    res[k] = {"content": m.get("content"), "reasoning": m.get("reasoning_content"), "s": round(time.time() - t, 2)}
res = {}
if mode == "serial":
    for k in P: ask(k)
else:
    ts = [threading.Thread(target=ask, args=(k,)) for k in P]; [t.start() for t in ts]; [t.join() for t in ts]
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items(): print(f"{k:7} {v['s']:7}s {repr((v['content'] or '')[:70])}")
EOF
U=http://127.0.0.1:8011/v1/chat/completions M=glm-5.3-flash-mia
python3 /tmp/glm_check.py $U $M /tmp/A.json serial            # after launch A; B -> /tmp/B.json; C -> both below
python3 /tmp/glm_check.py $U $M /tmp/C_serial.json serial
python3 /tmp/glm_check.py $U $M /tmp/C_conc.json concurrent
python3 - <<'EOF'
import json; f = ["/tmp/A.json", "/tmp/B.json", "/tmp/C_serial.json", "/tmp/C_conc.json"]
r = [json.load(open(x)) for x in f]
for k in r[0]:
    same = all((x[k]["content"], x[k]["reasoning"]) == (r[0][k]["content"], r[0][k]["reasoning"]) for x in r)
    print(f"{k:7} {'EXACT' if same else 'DIFFERS'}")
EOF
```

Pass:
1. **Greedy answers right** in A: `391`; 2, 3, 5, ..., 29; a working `fib`; 1 to 60; Tokyo / Japan; `4417` (the needle
   prompt is ~20K tokens: 4,096-row chunks in two micro-batch halves, row-split, own-row DSA fronts, sparse DSA past
   2,051 tokens); `red blue`. Same answers as production's engine on the same pack (bits may differ: other kernels).
2. **Drafted == serial, concurrent == serial**: every prompt `EXACT` across A, B, C serial and C concurrent (content and
   reasoning, byte for byte). Mia's contract plus ours: row-independent NVFP4 / 8-bit kernels, fp32 partials summed
   rank 0 first. A difference only on `needle` points at the prompt path (multi-prompt chunks, split); only on `vision`
   at the image rows; only in C at the batched verify window.
3. **Vision** right in A and C (rank 0 logs the tower; the reply names red, then blue).
4. **FP8 KV** on (log line), and C's pool line shows millions of tokens; optional: the 498,635-token prompt from the
   baseline harness (`bench/bench_tp2_night.py`) in A.
5. Speed, informative: single stream (B) near the 2026-10-11 baseline (count 110.6, code 80.6 tok/s, prefill 864 tok/s
   at 30K); C aggregate over 8 requests well past one stream's.

## 4. If it fails

- Load errors name the tensor. `PYTHONPATH=/srv/glm53-tf-mia/src python3 -m pytest
  /srv/glm53-tf-mia/tests/test_glm_modelopt_cpu.py` in the image (if it has pytest) checks the reader on the host; a
  split error on a real tensor name means the pack has a layout that test does not cover.
- Exact everywhere but with TF_GLM_SIDE, TF_GLM_L2PF or the split on: set that one to 0 in the copy and rerun A and C.

## 5. Restore production

`NAME=tfmia deploy/tf_tp4_mia.sh stop`, then the usual `deploy/tf_tp4.sh serve` (`glm-5.3-flash` on :8000).
