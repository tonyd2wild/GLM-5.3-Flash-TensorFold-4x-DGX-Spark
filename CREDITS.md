# Credits

**Special credit:** [tonyd2wild](https://github.com/tonyd2wild) (the 4× DGX Spark serving recipe) and
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) (the engine).

| who | what we use |
|---|---|
| **ashhart** ([TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) and its contributors | the engine (Python line, `python-0.6`) and its GLM-5.3-Flash CUDA family |
| **MiaAI-Lab** ([GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold), branch `tp4` @ `d220030`, Apache-2.0) | the multi-stream GLM engine in `engine-multi/`: 106 patches on TensorFold v0.6.0 (concurrent requests, TP-N, FP8 KV, RoCE all-gathers, split prefill, copy drafts, GLM vision); our derisked-weights loader is ported on top |
| **Brandon M. Music** ([brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw), ShapleyMCG License 1.0) | the EXL3 checkpoint we serve: routed experts as 4-bit `mcg` trellis, the rest as the official weights |
| **turboderp** ([ExLlamaV3](https://github.com/turboderp-org/exllamav3)) | the EXL3 format the checkpoint and TensorFold's EXL3 kernels follow |
| **incoai** (GLM-5.3-Flash-DFlash2, CC BY-NC-ND 4.0) | the DFlash2 drafter (read from our own copy, not redistributed) |
| **Z.ai** (zai-org/GLM-5.3-Flash) | the model |
| **NVIDIA** (nvidia/GLM-5.3-Flash-NVFP4, ModelOpt) | the NVFP4 base checkpoint our pack derives from |
| **knapcio** ([GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4), MIT) | the lossless8 conversion that produced our served pack |
| **Blackfrost** | the derisked GLM-5.3-Flash weights (our NVFP4 comparison pack) |
| **vLLM project** (Apache-2.0) | reference for the GLM-5.3-Flash vision tower and weight layout |
| **tonyd2wild** | fleet recipe, NFS layout, benchmark harness |

Add a row the day any idea, kernel or line of code from someone else is adopted.
