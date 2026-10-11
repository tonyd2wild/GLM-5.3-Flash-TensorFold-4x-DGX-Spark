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
