# Credits

**Special credit:** [tonyd2wild](https://github.com/tonyd2wild) (the Spark serving recipe this fleet runs),
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) (the engine) and Brandon M. Music's EXL3 checkpoint.

| who | what we use |
|---|---|
| **ashhart** ([TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) and its contributors | the engine (Python line, `python-0.6`) and its GLM-5.3-Flash CUDA family |
| **MiaAI-Lab** ([GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold), branch `tp4` @ `d220030`, Apache-2.0; their NOTICE kept verbatim in `engine-multi/NOTICE-MiaAI-Lab`) | the multi-stream GLM engine in `engine-multi/`: 106 patches on TensorFold v0.6.0 (concurrent requests, TP-N, FP8 KV, RoCE all-gathers, split prefill, copy drafts, GLM vision, 4-bit dense layers), and the two-Spark defaults the TP2 lane uses; our loader and changes are ported on top |
| **Contributors to MiaAI-Lab's patch series** (as credited in its README) | patches inside `engine-multi/`: [ezoushen](https://github.com/ezoushen) (a shared system prompt keeps each conversation's history; the display reservation in the pool; the pool compacting before it evicts), [desy0305](https://github.com/desy0305) (queued requests whose client left dropped at once), [meleesciony](https://github.com/meleesciony) (the kept-state cap fix, diagnosed and proposed), [Thomas Wade](https://github.com/ThomasWadeZ) (pictures read once, quoted media markers, kept-state limits), [johnwhited](https://github.com/johnwhited) (capacity refusals), [Eduardo Florencio](https://github.com/eduffd) (the Anthropic Messages API, backported from upstream work by evilpsycho42, ashhart and [Jordi Posthumus](https://github.com/JordiPosthumus)), [BadAd84](https://github.com/BadAd84) (long-context decode and prefill), [Robert Wojciechowski](https://github.com/wojo) (the spill tier for kept prompt states), [Lukasz Raczylo](https://github.com/lukaszraczylo) (the expert decode launch order), [jdecker76](https://github.com/jdecker76) (the reasoning-effort line, the kept-state count as a share of the pool) |
| **Jay Leaton** ([jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark), Apache-2.0) | patches adapted inside `engine-multi/`: the L2 prefetch during each layer's all-gathers (0460, `TF_GLM_L2PF`), the EXL3 decode kernel's 16-byte loads (0580, `TF_GLM_EXL3_LOADS`), tool-call and kept-reasoning rules (0620), part of the spill tier (0250) |
| **local-inference-lab** ([b12x](https://github.com/local-inference-lab/b12x), Apache-2.0) | the "RoCEnante" one-shot RoCE all-gather behind `TF_GLM_COMM=roce`: their C proxy (`roce_proxy.c`) and a GPU kernel rewritten from theirs (`roce.cu`) |
| **Brandon M. Music** ([ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg), ShapleyMcg License v1.0) | the EXL3 checkpoint we serve, [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw), produced with ShapleyMcg: routed experts as 4-bit `mcg` trellis, the rest as the official weights |
| **turboderp** ([ExLlamaV3](https://github.com/turboderp-org/exllamav3)) | the EXL3 format the checkpoint and TensorFold's EXL3 kernels follow |
| **incoai** (GLM-5.3-Flash-DFlash2, CC BY-NC-ND 4.0) | the DFlash2 drafter (read from our own copy, not redistributed) |
| **Z.ai** (zai-org/GLM-5.3-Flash) | the model |
| **NVIDIA** (nvidia/GLM-5.3-Flash-NVFP4, ModelOpt) | the NVFP4 base checkpoint our comparison pack derives from, and the checkpoint the vLLM reference runs |
| **knapcio** ([GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4), MIT) | the lossless8 conversion of our NVFP4 comparison pack, and the vLLM stack our reference numbers come from |
| **Blackfrost** | the derisked GLM-5.3-Flash weights (our NVFP4 comparison pack) |
| **vLLM project** (Apache-2.0) | the reference engine in our comparisons; reference for the GLM-5.3-Flash vision tower and weight layout |
| **tonyd2wild** | fleet recipe, NFS layout, launcher, benchmark harness and quality check |

Add a row the day any idea, kernel or line of code from someone else is adopted.

## ShapleyMcg attribution notice

Required by the ShapleyMcg License v1.0 (Section 3.2, Schedule B) for the benchmarks and documentation here, which use a checkpoint produced with it:

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.
