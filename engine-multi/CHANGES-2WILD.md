# What 2Wild Studios changed in this tree

`engine-multi/` is **TensorFold v0.6.0** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0)
with **MiaAI-Lab's GLM-5.3-Flash patch series** applied:
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
branch `tp4` @ `d220030`. That's 106 patches, about 8,000 changed lines in 93 files: concurrent requests, TP-N, FP8 KV, RoCE
all-gathers, split prefill, copy drafts, vision and 4-bit dense layers. Their NOTICE is kept verbatim in
[`NOTICE-MiaAI-Lab`](NOTICE-MiaAI-Lab).

The multi-stream GLM engine is their work. Everything we changed on top of it is listed below; nothing else differs.
You can check this by applying their `patches/*.patch` to TensorFold v0.6.0's `src/` with `patch -p0` (all 106 apply
cleanly) and running `diff -ru` against `engine-multi/src`. Each file below also starts with a one-line notice saying it
was changed. That line is not counted here.

| file | lines (+ / −) | what we changed |
|---|---|---|
| `tensorfold/families/glm5_next/cuda/modelopt.py` | **new, 166** | reader for ModelOpt mixed-precision GLM-5.3-Flash checkpoints: NVFP4 routed experts, MXFP8 and FP8-block projections as stored, BF16 elsewhere (ported from our earlier TP4 engine) |
| `tensorfold/families/glm5_next/cuda/forward.py` | +45 / −16 | ModelOpt projections and NVFP4 experts in the forward pass |
| `tensorfold/families/glm5_next/cuda/weights.py` | +36 / −17 | loading ModelOpt tensors (8-bit projections, NVFP4 experts, per-matrix scales) |
| `tensorfold/families/glm5_next/cuda/app.py` | +14 / −4 | vision on chat templates that write their own media markers (`emit_image` / `emit_video` get each part's `tf_mark`) |
| `tensorfold/families/glm5_next/cuda/split.py` | +13 / −7 | FP8 / E8M0 dtypes and scalar scales when splitting a ModelOpt checkpoint across ranks; pack markers dropped |
| `tensorfold/families/glm5_next/__init__.py` | +12 / −5 | `modelopt` accepted as a CUDA quant method, with the family check |
| `tensorfold/cuda/geometry.py` | +11 / −3 | memory geometry for ModelOpt checkpoints; scalar scales replicated |
| `tensorfold/families/glm5_next/cuda/qmm.py` | +10 / −1 | ModelOpt 8-bit projections run their own matmul |
| `tensorfold/families/glm5_next/cuda/engine.py` | +6 / −3 | ModelOpt checkpoints take the same draft policy as EXL3; load summary |
| `tensorfold/cuda/capacity.py` | +4 / −3 | capacity count reads each tensor from the shard the index names (skips stale copies) |
| **total** | **+317 / −59 in 10 files** | |

The ModelOpt reader is only used for the NVFP4 comparison lane. On the **EXL3 lanes this repo serves by default**, the
only changes that do anything are the vision-template fix (`app.py`) and a few loader guards: scalar scales, stale shard
copies and pack markers. Everything else is MiaAI-Lab's code as published.
