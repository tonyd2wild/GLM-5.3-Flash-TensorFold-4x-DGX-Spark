# TP4 lane, the default: EXL3 with 4-bit dense layers (2026-10-11)

Checkpoint: Brandon M. Music's [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
(EXL3 4-bit `mcg` routed experts, the rest as the official BF16 weights; 164 GiB), served with `TF_GLM_DENSE=q4`: the
checkpoint's BF16 non-expert matrices (attention, indexer, shared and dense MLP) quantized at load to 4-bit affine groups
of 64 with MSE-searched ranges, the head in FP8, `kv_b`, embedding and vision tower BF16. Engine `engine-multi/`, TP4 on
4× DGX Spark (Reddie rank 0, weights over NFS), FP8 KV, RoCE all-gathers, MTP + DFlash2 + copy drafts, vision on,
`reasoning_effort: low`, temperature 0. Launch: [`deploy/serve_exl3.sh`](../../deploy/serve_exl3.sh) (`TP=4`, the default).

| | |
|---|---|
| max context | 500,000 tokens a request (prompt + reply) |
| streams | 8 at once |
| KV pool | 5,285,888 tokens shared (FP8): all 8 streams can hold 500K at once (4,000,128), 1.29M left for kept prompt states |
| memory | startup estimate 48.6 of 91.6 GiB a Spark |

Run on a private port (`:8001`) so fleet traffic could not share the streams; 23:41-00:10 UTC, no preemptions, no
failures. Harness: [`bench/bench_tp2_night.py`](../../bench/bench_tp2_night.py) `--suite baseline --prefill-sizes
2048,8192,32768,131072`. Raw: [`results/`](results/). Charts and the scripts that draw them: [`charts/`](charts/)
(`tf-vs-vllm-*` is the one-image comparison, `hero_chart.py`).

"TensorFold (FP8 dense)" is the same checkpoint with `TF_GLM_DENSE=fp8` the same evening
([`../2026-10-11-exl3/`](../2026-10-11-exl3/TABLES.md)); "vLLM knapcio stack" is the fleet's vLLM on
`nvidia/GLM-5.3-Flash-NVFP4` (2026-09-29, same harness, 500K, effort low).

**Quality check** ([`bench/qcheck.py`](../../bench/qcheck.py), greedy, 10 short prompts with known answers including
French, Spanish and Chinese, where MiaAI-Lab reported q4 can lose the end of turn): 10 / 10 replies end cleanly and are
right, as with FP8 dense ([`results/qcheck-exl3-q4dense-tp4.json`](results/qcheck-exl3-q4dense-tp4.json)).

### Single stream (decode tok/s, median of 3; time to first token)

| prompt | TensorFold (q4 dense) | TTFT | vs vLLM | TensorFold (FP8 dense) | vLLM knapcio stack |
|---|---|---|---|---|---|
| count to 100 | **180.2** | 0.11 s | 99 % | 163.7 | 181.7 |
| count to 300 | **172.8** | 0.11 s | 104 % | 157.3 | 165.7 |
| code | **127.0** | 0.19 s | 105 % | 101.4 | 120.5 |
| json | **132.7** | 0.11 s | 110 % | 119.5 | 120.5 |
| sql | **113.5** | 0.11 s | 104 % | 84.9 | 109.1 |
| tool call ¹ | **93.2** | 0.24 s | 137 % | 88.7 | 67.9 |
| math | **121.2** | 0.07 s | 85 % | 115.8 | 141.8 |
| structured (YAML) | **117.6** | 0.12 s | – | 77.8 | – |
| summary | **78.5** | 0.17 s | 111 % | 73.4 | 70.6 |
| prose | **65.4** | 0.17 s | 107 % | 54.1 | 61.0 |
| narrative | **64.9** | 0.17 s | 107 % | 57.9 | 60.8 |

¹ end to end: the server sends a tool call as one chunk when it completes.

### C1–C6 by prompt type (aggregate tok/s, median of 3 rounds / per-stream decode / TTFT p50 · p90)

| type | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| prose | **65.2** / 66.9 / 0.29·0.29 s | **92.8** / 48.7 / 0.51·0.54 s | **113.7** / 39.1 / 0.37·0.54 s | **130.6** / 33.8 / 0.39·0.40 s | **141.8** / 29.4 / 0.42·0.60 s | **152.2** / 26.8 / 0.45·0.62 s |
| code | **126.4** / 133.9 / 0.32·0.33 s | **166.4** / 88.0 / 0.36·0.54 s | **205.3** / 72.5 / 0.40·0.59 s | **218.3** / 57.9 / 0.62·0.63 s | **242.1** / 50.7 / 0.54·0.67 s | **255.7** / 46.4 / 0.71·0.76 s |
| math | **111.2** / 118.8 / 0.31·0.31 s | **158.7** / 88.3 / 0.48·0.54 s | **198.6** / 70.1 / 0.44·0.57 s | **226.3** / 61.2 / 0.46·0.65 s | **248.0** / 53.7 / 0.50·0.68 s | **270.7** / 48.0 / 0.53·0.72 s |
| json | **125.5** / 133.0 / 0.35·0.36 s | **172.0** / 92.7 / 0.46·0.59 s | **229.5** / 80.6 / 0.49·0.69 s | **265.6** / 69.6 / 0.52·0.52 s | **266.5** / 56.7 / 0.58·0.77 s | **301.4** / 53.1 / 0.60·0.80 s |
| count to 100 | **158.2** / 176.2 / 0.14·0.14 s | **235.7** / 137.4 / 0.22·0.28 s | **300.1** / 117.3 / 0.23·0.36 s | **348.4** / 97.9 / 0.25·0.25 s | **382.6** / 86.2 / 0.27·0.40 s | **438.7** / 81.2 / 0.29·0.30 s |
| structured (YAML) | **93.7** / 115.9 / 0.36·0.36 s | **156.2** / 99.0 / 0.46·0.46 s | **203.7** / 83.8 / 0.49·0.49 s | **223.1** / 73.4 / 0.67·0.72 s | **258.6** / 62.6 / 0.56·0.57 s | **287.5** / 58.0 / 0.60·0.60 s |

### C1–C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)

| streams | TensorFold (q4 dense) | TTFT p90 | TensorFold (FP8 dense) | vLLM knapcio stack |
|---|---|---|---|---|
| C1 | **124.0** / 124.2 | 0.3 s | 114.1 / 119.0 | 115.4 / 116.5 |
| C2 | **106.0** / 158.2 | 0.6 s | 97.3 / 152.8 | 103.4 / 152.6 |
| C3 | **117.0** / 174.7 | 0.5 s | 107.0 / 168.4 | 104.5 / 155.8 |
| C4 | **175.8** / 176.6 | 0.5 s | 170.2 / 170.2 | 159.9 / 162.1 |
| C5 | **143.4** / 189.0 | 0.6 s | 135.7 / 184.2 | 134.5 / 176.1 |
| C6 | **151.2** / 166.2 | 0.6 s | 141.1 / 146.5 | 141.2 / 146.5 |

### Cold prefill (salted prompts, one output token)

| prompt | TensorFold (q4 dense) | TTFT | TensorFold (FP8 dense) | vLLM knapcio stack |
|---|---|---|---|---|
| 5,942 tokens | **2,447 tok/s** | 2.4 s | 2,230 tok/s | 2,378 tok/s |
| 29,866 tokens | **2,780 tok/s** | 10.7 s | 2,536 tok/s | 2,798 tok/s |
| 113,911 tokens | **2,808 tok/s** | 40.6 s | 2,561 tok/s | 2,794 tok/s |
| 498,635 tokens | **2,611 tok/s** | 190.9 s | 2,398 tok/s | 2,462 tok/s |

**Vision check:** PASS in 1.46 s: "The image shows red on the left side and blue on the right side."

