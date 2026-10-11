# Multi-stream engine: full baseline (2026-10-11)

Engine `engine-multi/` (TensorFold v0.6.0 + MiaAI-Lab's GLM patches, Apache-2.0, + our ModelOpt port and vision
template fix), TP4 on 4× DGX Spark, derisked lossless8 weights over NFS from Reddie, `--parallel 8`, FP8 KV
(`TF_GLM_KV=fp8`), RoCE all-gathers, MTP + DFlash2 + copy drafts, 500K window per stream, vision on,
`reasoning_effort: low`, temperature 0. Launch: `NAME=tfmia PROFILE=multi deploy/tf_tp4.sh serve`.

Run on a private port (`:8001`) so fleet traffic could not share the 8 streams; 22:17-22:52 UTC, no preemptions,
no failures. Harness: [`bench/bench_tp2_night.py`](../../bench/bench_tp2_night.py) `--suite baseline
--prefill-sizes 2048,8192,32768,131072`. Raw: [`results/`](results/). Charts and the scripts that draw them:
[`charts/`](charts/) (`python3 build_report_multi.py MULTI.json SINGLE.json VLLM.json OUT && python3
make_charts.py OUT/charts.json OUT`).

The vLLM column is the fleet's knapcio stack on `nvidia/GLM-5.3-Flash-NVFP4` (2026-09-29, same harness, 500K, effort
low); "one-at-a-time build" is our `engine/` baseline from the same day
([`../2026-10-11-baseline/`](../2026-10-11-baseline/)).

### Single stream (decode tok/s, median of 3; time to first token)

| prompt | multi-stream engine | TTFT | vs vLLM | vLLM knapcio stack | our one-at-a-time build |
|---|---|---|---|---|---|
| count to 100 | **149.9** | 0.15 s | 82 % | 181.7 | 110.6 |
| count to 300 | **141.8** | 0.15 s | 86 % | 165.7 | 106.3 |
| code | **109.0** | 0.16 s | 90 % | 120.5 | 80.6 |
| json | **113.0** | 0.13 s | 94 % | 120.5 | 81.4 |
| sql | **83.4** | 0.08 s | 76 % | 109.1 | 64.9 |
| tool call ¹ | **82.5** | 0.25 s | 121 % | 67.9 | 82.7 |
| math | **106.6** | 0.12 s | 75 % | 141.8 | 78.2 |
| structured (YAML) | **103.2** | 0.13 s | – | – | 74.9 |
| summary | **63.6** | 0.17 s | 90 % | 70.6 | 55.7 |
| prose | **54.4** | 0.18 s | 89 % | 61.0 | 46.8 |
| narrative | **54.6** | 0.18 s | 90 % | 60.8 | 45.7 |

¹ end to end: the server sends a tool call as one chunk when it completes.

### C1–C6 by prompt type (aggregate tok/s, median of 3 rounds / per-stream decode / TTFT p50 · p90)

| type | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| prose | **52.9** / 53.9 / 0.25·0.25 s | **79.9** / 41.2 / 0.40·0.46 s | **96.2** / 34.0 / 0.48·0.53 s | **114.4** / 29.2 / 0.37·0.40 s | **124.9** / 25.8 / 0.38·0.45 s | **131.9** / 23.2 / 0.48·0.58 s |
| code | **100.0** / 104.5 / 0.31·0.31 s | **142.8** / 76.5 / 0.52·0.52 s | **175.9** / 63.7 / 0.40·0.56 s | **189.0** / 50.4 / 0.44·0.59 s | **216.7** / 45.3 / 0.52·0.66 s | **229.5** / 40.7 / 0.55·0.56 s |
| math | **95.8** / 102.2 / 0.34·0.35 s | **139.3** / 76.8 / 0.48·0.57 s | **166.3** / 61.8 / 0.62·0.63 s | **192.6** / 53.3 / 0.64·0.69 s | **206.3** / 48.3 / 0.57·0.59 s | **228.6** / 42.7 / 0.80·0.82 s |
| json | **102.0** / 106.9 / 0.35·0.35 s | **153.3** / 80.7 / 0.44·0.44 s | **189.9** / 67.9 / 0.66·0.67 s | **211.6** / 58.4 / 0.53·0.71 s | **242.2** / 53.0 / 0.58·0.58 s | **265.9** / 47.4 / 0.62·0.62 s |
| count to 100 | **133.3** / 146.1 / 0.14·0.18 s | **194.9** / 110.1 / 0.24·0.33 s | **252.4** / 96.0 / 0.23·0.28 s | **298.2** / 89.0 / 0.42·0.43 s | **333.0** / 76.4 / 0.48·0.49 s | **365.7** / 69.5 / 0.36·0.58 s |
| structured (YAML) | **85.1** / 100.0 / 0.35·0.36 s | **131.4** / 81.3 / 0.44·0.58 s | **163.4** / 63.5 / 0.48·0.49 s | **174.5** / 55.3 / 0.66·0.70 s | **214.8** / 51.7 / 0.58·0.76 s | **237.5** / 47.7 / 0.62·0.87 s |

### C1–C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)

| streams | multi-stream engine | TTFT p90 | vLLM knapcio stack | our one-at-a-time build |
|---|---|---|---|---|
| C1 | **99.9** / 103.2 | 0.3 s | 115.4 / 116.5 | 74.6 / 75.4 |
| C2 | **84.9** / 134.1 | 0.5 s | 103.4 / 152.6 | 67.2 / 77.2 |
| C3 | **100.6** / 151.9 | 0.6 s | 104.5 / 155.8 | 56.1 / 75.3 |
| C4 | **154.7** / 155.8 | 0.5 s | 159.9 / 162.1 | 74.5 / 74.7 |
| C5 | **125.8** / 166.5 | 0.8 s | 134.5 / 176.1 | 57.3 / 75.9 |
| C6 | **134.4** / 136.0 | 0.7 s | 141.2 / 146.5 | 61.1 / 66.1 |

### Cold prefill (salted prompts, one output token)

| prompt | multi-stream engine | TTFT | vLLM knapcio stack | our one-at-a-time build |
|---|---|---|---|---|
| 5,943 tokens | **1,675 tok/s** | 3.5 s | 2,378 tok/s | 864 tok/s |
| 29,867 tokens | **1,984 tok/s** | 15.1 s | 2,798 tok/s | 894 tok/s |
| 113,911 tokens | **2,038 tok/s** | 55.9 s | 2,794 tok/s | 887 tok/s |
| 498,635 tokens | **1,928 tok/s** | 258.7 s | 2,462 tok/s | 772 tok/s |

**Vision check:** PASS in 1.59 s: "The image shows red on the left half and blue on the right half."

