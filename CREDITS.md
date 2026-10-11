# Credits

**Special credit:** [tonyd2wild](https://github.com/tonyd2wild) (the 4× DGX Spark serving recipe) and
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) (the engine).

| who | what we use |
|---|---|
| **ashhart** ([TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) and its contributors | the engine (Python line, `python-0.6`) and its GLM-5.3-Flash CUDA family |
| **Z.ai** (zai-org/GLM-5.3-Flash) | the model |
| **NVIDIA** (nvidia/GLM-5.3-Flash-NVFP4, ModelOpt) | the NVFP4 base checkpoint our pack derives from |
| **knapcio** ([GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4), MIT) | the lossless8 conversion that produced our served pack |
| **Blackfrost** | the derisked GLM-5.3-Flash weights |
| **vLLM project** (Apache-2.0) | reference for the GLM-5.3-Flash vision tower and weight layout |
| **tonyd2wild** | fleet recipe, NFS layout, benchmark harness |

Add a row the day any idea, kernel or line of code from someone else is adopted.
