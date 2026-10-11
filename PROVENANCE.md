# Provenance

Where every part of this build comes from. Updated with each change.

## Base

| part | source | license | notes |
|---|---|---|---|
| engine | ashhart/TensorFold, branch `python-0.6` @ `ed78d6f` (2026-10-07) | Apache-2.0 | the Python engine; GLM-5.3-Flash CUDA family `families/glm5_next/cuda/` |

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
| benchmark harness `bench/bench_tp2_night.py` | our own (tonyd2wild speed-night harness) | ours |

## References consulted (no code copied unless listed above)

| reference | license | used for |
|---|---|---|
| vLLM `models/glm5next/nvidia/` (in our v11 image) | Apache-2.0 | GLM-5.3-Flash vision tower and weight layout |
| knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 | MIT | the lossless8 checkpoint layout we serve |
