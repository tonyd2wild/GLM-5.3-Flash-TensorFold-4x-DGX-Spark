# Multi-stream engine on our weights: count-to-100 test (2026-10-11)

Engine: `engine-multi/` (TensorFold v0.6.0 + MiaAI-Lab's GLM patches, Apache-2.0, + our ModelOpt port), TP4, derisked
lossless8 over NFS, `--parallel 8`, FP8 KV (`TF_GLM_KV=fp8`), RoCE all-gathers (b12x RoCEnante transport, up to 1 MiB),
DFlash2 drafts, 500K window per stream (shared pool of 5,285,888 tokens), `reasoning_effort: low`, temperature 0,
vision off (her vision patch does not recognize this checkpoint's chat template yet). Startup: one-row verify 25.1 ms
(our previous build 29.0 ms), estimate 53.3 of 90.2 GiB a rank.

| count to 100 | previous build (`engine/`) | multi-stream engine | vLLM knapcio (reference) |
|---|---|---|---|
| 1 at a time (per stream) | 110.6 tok/s | **146.2 / 148.2** | 181.7 |
| 4 at once (aggregate) | ~95 (queued, TTFT up to 6.6 s) | **266.7 / 279.8** (TTFT ≤ 0.6 s) | |
| 8 at once (aggregate) | ~95 (queued) | **374.8 / 382.4** (TTFT ≤ 0.74 s) | |

Two rounds each. Greedy answer checked (capital of France; first five primes).

## Quick comparison against vLLM (same harness, same day)

Single stream (decode tok/s, best of 2):

| prompt | multi-stream engine | previous build | vLLM knapcio (reference) |
|---|---|---|---|
| code | 106.0 | 80.6 | 120.5 |
| json | 112.8 | 81.4 | 120.5 |
| math | 104.1 | 78.2 | 141.8 |
| sql | 94.0 | 64.9 | 109.1 |
| summary | 62.7 | 55.7 | 70.6 |
| prose | 53.6 | 46.8 | 61.0 |
| count to 300 | 144.1 | 106.3 | 165.7 |

Mixed real prompts, C1–C6 (aggregate tok/s, median / peak of 2 rounds; TTFT p90):

| streams | multi-stream engine | vLLM knapcio (median / peak, 3 rounds) |
|---|---|---|
| C1 | 102.0 / 103.6 (0.33 s) | 115.4 / 116.5 |
| C2 | **107.4** / 131.3 (0.78 s) | 103.4 / 152.6 |
| C3 | **112.8** / 154.1 (0.50 s) | 104.5 / 155.8 |
| C4 | 131.6 / 155.8 (0.46 s) | 159.9 / 162.1 |
| C5 | **147.1** / 168.5 (0.60 s) | 134.5 / 176.1 |
| C6 | 132.5 / 135.7 (0.86 s) | 141.2 / 146.5 |

Code alone (700 tokens; aggregate tok/s, per stream, max TTFT; 2 rounds):

| at once | multi-stream engine | vLLM knapcio conc_bench (768 tokens, its harness) |
|---|---|---|
| 1 | 98–100 (103 per stream, 0.3 s) | 117 |
| 2 | **138–141** (74 per stream, 0.4 s) | 134 |
| 4 | **193–197** (51 per stream, 0.6 s) | 182 |
| 6 | **228–231** (40 per stream, 0.6 s) | – |
| 8 | 197 (30 per stream, 0.6 s); later rounds shared the 8 slots with fleet traffic (44–48 s TTFT) | 232 |

The engine serves `:8000`, so fleet agents' requests share its 8 streams: `/health` showed three fleet prompts
prefilling when the C8 rounds ran. The full baseline will run on a private port.
