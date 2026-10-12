# Provenance

Where every part of this build comes from. Updated with each change.

## Base

| part | source | license | notes |
|---|---|---|---|
| engine | ashhart/TensorFold, branch `python-0.6` @ `ed78d6f` (2026-10-07) | Apache-2.0 | the Python engine; GLM-5.3-Flash CUDA family `families/glm5_next/cuda/` |

## `engine-multi/` (the serving engine since 2026-10-11)

| part | source | license | notes |
|---|---|---|---|
| engine | ashhart/TensorFold v0.6.0 | Apache-2.0 | the Python engine |
| GLM patches | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, branch `tp4` @ `d220030` (106 patches) | Apache-2.0 | concurrent requests, TP-N, FP8 KV, RoCE all-gathers, split prefill, copy drafts, vision; kept as published, their NOTICE verbatim in `engine-multi/NOTICE-MiaAI-Lab`, our changes on top listed in `engine-multi/CHANGES-2WILD.md` |
| ModelOpt reader, splitter, family check; NVFP4 experts and 8-bit linears in every forward; CPU test; `TEST.md` | ours | ours, on Apache-2.0 code | ports our derisked pack onto the engine |
| vision on templates that write their own media markers (`media_template`) | ours | ours, on Apache-2.0 code | |

Updated with `git subtree pull --prefix=engine-multi <tree> mia-derisk --squash`.

## `engine/` (the previous one-at-a-time build): removed 2026-10-12

The first, one-request-at-a-time build was replaced by `engine-multi/` on 2026-10-11 and the `engine/` tree was removed from the repo on 2026-10-12. It stays in the git history. It was the TensorFold Python line (ashhart, Apache-2.0) with our GLM-5.3-Flash TP4 / ModelOpt / vision work. We had removed one third-party author's code from it, and their license file and mentions went with that code. That author is MiaAI-Lab, and their code is the engine this repo now serves (see above). The benchmark rows labelled "first build" were measured on that tree.

## Removed from the base

Every line whose commit was authored solely by one excluded third-party author, found with `git blame` against the
15 commits below (1,058 lines in 32 files on `python-0.6`), plus code adapted from that author's patches, their
license file and every mention of them. Commits authored by others are kept.

| commit | subject | handling |
|---|---|---|
| 7c088eb | DFlash2 block attention over its sliding window (ring) | reverted to the whole-window draft buffer |
| 17e90a4 | `TF_GLM_MTP=0\|1\|auto` | switch removed; MTP head loaded as before |
| b23c10a | prompt chunks' dense latent pass and pool scores 512 rows at a time | reverted |
| 3f35ea9 | startup estimate counts EXL3 experts' scratch | reverted |
| 358875c | idle rank 1 waits on the rendezvous TCP store | reverted |
| 93d6462 | `TENSORFOLD_MEMORY_RESERVE_GIB` | removed |
| 51b098d | live token totals on the CUDA `/health` | removed |
| 8f5e680 | `/tokenize` and `/detokenize` | removed |
| ad15127 | `chat_template_kwargs.thinking`; GLM keeps earlier-turn reasoning | reverted |
| b3d8733 | `reasoning_effort max` mapping | reverted |
| 588921b, aaecbde, c2b9045, 69d0ca9, fe2b514 | Qwen / Flash Next vision, video, image budget, tool parameters | removed with the Qwen families |

## Added by us

| part | written from | license of the reference |
|---|---|---|
| `families/glm5_next/cuda/modelopt.py` (ModelOpt checkpoint reader: NVFP4 experts, MXFP8 / FP8-block / BF16 linears) | the engine's own NVFP4 module API; the checkpoint's tensor layout | ours, on Apache-2.0 code |
| `vision/glm_cuda.py` (vision tower, processor, prompt expansion) | vLLM `models/glm5next/nvidia/multimodal.py` and `transformers_utils/processors/glm5next.py` | Apache-2.0 (vLLM contributors) |
| engine wiring: TP4 rank checks, image rows across ranks, NVFP4 MoE dispatch, 8-bit linears, index-aware capacity scan, split rules for scalars / FP8 / pack markers | ours | ours, on Apache-2.0 code |
| `deploy/tf_tp4.sh` | ours (fleet NCCL / RoCE settings from our live deployment) | ours |
| `deploy/serve_exl3.sh` (TP4 / TP2 lanes), `bench/qcheck.py` (the quality check), `runs/*/charts/*.py` (tables and charts) | ours | ours |
| the TP2 lane's settings in `deploy/tf_tp4.sh` (streams, kept-state pool, reserve) | MiaAI-Lab's two-Spark defaults (Apache-2.0) | Apache-2.0 |
| benchmark harness `bench/bench_tp2_night.py` | our own (tonyd2wild speed-night harness) | ours |

## References consulted (no code copied unless listed above)

| reference | license | used for |
|---|---|---|
| vLLM `models/glm5next/nvidia/` (in our v11 image) | Apache-2.0 | GLM-5.3-Flash vision tower and weight layout |
| knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 | MIT | the lossless8 checkpoint layout of our NVFP4 pack |
| brandonmusic/GLM-5.3-Flash-tr3-4bpw | ShapleyMCG License 1.0 | the EXL3 checkpoint served (read as published, not redistributed) |
| ashhart/TensorFold `docs/recipes` (glm-5.3-flash, cuda, exl3) | Apache-2.0 | EXL3 over NVFP4 for GLM; GLM on CUDA serialized upstream |
