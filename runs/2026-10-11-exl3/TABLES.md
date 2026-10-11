# EXL3 checkpoint on the multi-stream engine: full baseline (2026-10-11)

Checkpoint: Brandon M. Music's [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
(EXL3 4-bit `mcg` routed experts, the rest as the official BF16 weights; 164 GiB), served with `TF_GLM_DENSE=fp8` (the
BF16 non-expert matrices held as FP8 with a scale per 128 block; `kv_b`, embedding and vision tower BF16). Engine
`engine-multi/`, TP4 on 4× DGX Spark, weights over NFS from Reddie, `--parallel 8`, FP8 KV, RoCE all-gathers, MTP +
DFlash2 + copy drafts, 500K window per stream, vision on, `reasoning_effort: low`, temperature 0. Launch:
[`deploy/serve_exl3.sh`](../../deploy/serve_exl3.sh). Startup: one-row verify 21.1 ms (NVFP4 derisked: 25.1 ms),
estimate 51.4 of 91.6 GiB a rank.

Run on a private port (`:8001`) so fleet traffic could not share the 8 streams; 23:04-23:35 UTC, no preemptions, no
failures. Harness: [`bench/bench_tp2_night.py`](../../bench/bench_tp2_night.py) `--suite baseline --prefill-sizes
2048,8192,32768,131072`. Raw: [`results/`](results/). Charts and the scripts that draw them: [`charts/`](charts/).

"NVFP4 derisked" is the same engine on our NVFP4 pack the same evening
([`../2026-10-11-multi-engine/`](../2026-10-11-multi-engine/TABLES.md)); "vLLM knapcio stack" is the fleet's vLLM on
`nvidia/GLM-5.3-Flash-NVFP4` (2026-09-29, same harness, 500K, effort low).

**Quality check** ([`bench/qcheck.py`](../../bench/qcheck.py), greedy, 10 short prompts with known answers including
French, Spanish and Chinese): 10 / 10 replies end cleanly and are right
([`results/qcheck-exl3-fp8dense.json`](results/qcheck-exl3-fp8dense.json)).

### Single stream (decode tok/s, median of 3; time to first token)

| prompt | EXL3 | TTFT | vs vLLM | NVFP4 derisked | vLLM knapcio stack |
|---|---|---|---|---|---|
| count to 100 | **163.7** | 0.11 s | 90 % | 149.9 | 181.7 |
| count to 300 | **157.3** | 0.11 s | 95 % | 141.8 | 165.7 |
| code | **101.4** | 0.19 s | 84 % | 109.0 | 120.5 |
| json | **119.5** | 0.11 s | 99 % | 113.0 | 120.5 |
| sql | **84.9** | 0.13 s | 78 % | 83.4 | 109.1 |
| tool call ¹ | **88.7** | 0.24 s | 131 % | 82.5 | 67.9 |
| math | **115.8** | 0.11 s | 82 % | 106.6 | 141.8 |
| structured (YAML) | **77.8** | 0.15 s | – | 103.2 | – |
| summary | **73.4** | 0.17 s | 104 % | 63.6 | 70.6 |
| prose | **54.1** | 0.21 s | 89 % | 54.4 | 61.0 |
| narrative | **57.9** | 0.18 s | 95 % | 54.6 | 60.8 |

¹ end to end: the server sends a tool call as one chunk when it completes.

### C1–C6 by prompt type (aggregate tok/s, median of 3 rounds / per-stream decode / TTFT p50 · p90)

| type | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| prose | **57.5** / 58.7 / 0.25·0.25 s | **84.7** / 44.0 / 0.43·0.46 s | **106.0** / 36.5 / 0.46·0.47 s | **122.0** / 31.3 / 0.50·0.51 s | **135.0** / 28.1 / 0.41·0.41 s | **143.7** / 25.3 / 0.44·1.02 s |
| code | **115.8** / 121.1 / 0.27·0.28 s | **170.1** / 90.0 / 0.44·0.47 s | **188.2** / 69.0 / 0.38·0.52 s | **232.9** / 62.1 / 0.55·0.55 s | **230.0** / 48.4 / 0.52·0.61 s | **265.5** / 46.9 / 0.52·0.53 s |
| math | **100.7** / 106.8 / 0.26·0.27 s | **149.1** / 79.2 / 0.32·0.32 s | **179.2** / 65.7 / 0.37·0.50 s | **204.5** / 55.2 / 0.43·0.55 s | **218.4** / 48.8 / 0.46·0.60 s | **256.5** / 46.1 / 0.49·0.50 s |
| json | **113.7** / 119.1 / 0.32·0.32 s | **174.9** / 93.2 / 0.52·0.52 s | **216.3** / 76.3 / 0.58·0.59 s | **239.2** / 62.8 / 0.48·0.63 s | **267.1** / 56.3 / 0.53·0.68 s | **301.9** / 52.7 / 0.73·0.74 s |
| count to 100 | **147.3** / 162.4 / 0.14·0.14 s | **232.2** / 131.4 / 0.18·0.28 s | **299.3** / 112.0 / 0.20·0.31 s | **347.7** / 94.6 / 0.23·0.23 s | **368.8** / 81.6 / 0.25·0.41 s | **430.4** / 80.3 / 0.28·0.36 s |
| structured (YAML) | **91.1** / 109.0 / 0.32·0.32 s | **140.5** / 84.0 / 0.39·0.53 s | **172.8** / 72.0 / 0.44·0.57 s | **207.2** / 63.3 / 0.48·0.48 s | **250.2** / 60.5 / 0.54·0.54 s | **255.0** / 50.1 / 0.58·0.58 s |

### C1–C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)

| streams | EXL3 | TTFT p90 | NVFP4 derisked | vLLM knapcio stack |
|---|---|---|---|---|
| C1 | **114.1** / 119.0 | 0.3 s | 99.9 / 103.2 | 115.4 / 116.5 |
| C2 | **97.3** / 152.8 | 0.5 s | 84.9 / 134.1 | 103.4 / 152.6 |
| C3 | **107.0** / 168.4 | 0.5 s | 100.6 / 151.9 | 104.5 / 155.8 |
| C4 | **170.2** / 170.2 | 0.4 s | 154.7 / 155.8 | 159.9 / 162.1 |
| C5 | **135.7** / 184.2 | 0.6 s | 125.8 / 166.5 | 134.5 / 176.1 |
| C6 | **141.1** / 146.5 | 0.8 s | 134.4 / 136.0 | 141.2 / 146.5 |

### Cold prefill (salted prompts, one output token)

| prompt | EXL3 | TTFT | NVFP4 derisked | vLLM knapcio stack |
|---|---|---|---|---|
| 5,942 tokens | **2,230 tok/s** | 2.7 s | 1,675 tok/s | 2,378 tok/s |
| 29,865 tokens | **2,536 tok/s** | 11.8 s | 1,984 tok/s | 2,798 tok/s |
| 113,909 tokens | **2,561 tok/s** | 44.5 s | 2,038 tok/s | 2,794 tok/s |
| 498,633 tokens | **2,398 tok/s** | 207.9 s | 1,928 tok/s | 2,462 tok/s |

**Vision check:** PASS in 1.58 s: "The image shows red on the left half and blue on the right half."

