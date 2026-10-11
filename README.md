# GLM-5.3-Flash on TensorFold, TP4 on 4× DGX Spark, with our derisked weights

> **Special credit:** [tonyd2wild](https://github.com/tonyd2wild) (the 4× Spark serving recipe this fleet runs) and
> [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (the engine). Full ledger: [CREDITS.md](CREDITS.md).

**Private working repo.** Goal: serve GLM-5.3-Flash on the TensorFold engine at tensor-parallel 4 across our four
DGX Sparks, from our derisked weights as stored, with Reddie as the head node serving the weights over NFS.

| target | value |
|---|---|
| weights | `glm-quant-mix-derisked/lossless8` (198 GB, on Reddie; NFS to the other three) |
| layout | TP4: Reddie (rank 0, HTTP), Spark4, Asusi, Bluey |
| context | 500,000 tokens |
| vision | on (images) |
| thinking | `reasoning_effort: low` by default |
| endpoint | `glm-5.3-flash` on `:8000` (the fleet's name and port) |
| baseline | C1–C6 × prose, code, math, JSON, count-to-100, structured: prefill, decode, TTFT, throughput |

## Approach

No TensorFold build serves GLM-5.3-Flash on CUDA at TP4, and none serves our checkpoint format. TensorFold's Python
engine (`python-0.6`, Apache-2.0, ashhart) serves GLM-5.3-Flash on CUDA at TP2 from MLX 4-bit weights, one request
at a time, with images on MLX only. We start from it and add:

1. **TP4** (today: two ranks).
2. **Our checkpoint as stored**: NVFP4 routed experts through the engine's existing NVFP4 kernels; the MXFP8 KDA
   projections, FP8-block shared experts and DSA projections dequantized to BF16 at load (about 3 GB a rank).
3. **Concurrent requests** for the C1–C6 sweep.
4. **Vision on CUDA**: GLM's vision tower in torch on rank 0, its features spliced into the prompt.
5. **500K context**, `reasoning_effort` low.

## Provenance policy

Code from one excluded third-party author is not used: their commits, code adapted from their patches, their license
file and every mention of them are removed. Commits authored by others are kept as part of the engine. Public facts
and measurements may inform the work; code is written by us or taken from other open-source authors with credit.
Every piece's origin is logged in [PROVENANCE.md](PROVENANCE.md).

## Status

| step | state |
|---|---|
| Remove the excluded author's lines, adapted code and mentions from the Python engine | done (0 lines left) |
| Map the GLM CUDA engine for TP4, our checkpoint, concurrency, vision | in progress |
| Benchmark harness (`bench/`) with the per-type C1–C6 sweep and a vision check | ready; verified against the live vLLM lane |
| TP4, loader, concurrency, vision | not started |
| Boot on all four Sparks (needs the live GLM down) | not started |
| Baseline | not started |

## Fleet facts

| node | rank | free disk (2026-10-11) |
|---|---|---|
| Reddie | 0 (head, NFS server, HTTP) | 235 GB |
| Spark4 | 1 | 47 GB |
| Asusi | 2 | 37 GB |
| Bluey | 3 | 559 GB |

Each node has about 20 GiB free memory while the live GLM serves, so GPU work waits for a window. The existing
`glm53-roce:v11` image already carries Python 3.12, torch 2.13 (CUDA 13.0), NCCL 2.29 and nvcc 13.0.
