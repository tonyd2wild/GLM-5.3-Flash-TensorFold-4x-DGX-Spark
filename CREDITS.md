# Credits

**Special credit:** [WestWaters/pollard-weights](https://github.com/WestWaters/pollard-weights) (the method, and the
home of our EXL3 expert data) and [tonyd2wild](https://github.com/tonyd2wild) (the 4× DGX Spark serving recipe) get a
named call-out at the top of every artifact from this work.

| who | what we use |
|---|---|
| **ashhart** ([TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) | the engine |
| **bertholomus** ([TensorFold fork, `deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4); [deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10), Apache-2.0) | the DeepSeek-V4.1 TP4 port, recipe and kernel kit we build on (`engine/`) |
| **jayleaton** (ashhart/TensorFold PRs #634-#674) | the upstream DeepSeek-V4.1 PR stack |
| **ecohash-co** (ashhart/TensorFold#559) | independent 4-node reproduction and the fixes we re-implement: RDMA GID override, `_index_score` variant fallback, prompt sequence-parallel rows |
| **DeepSeek-AI** (DeepSeek-V4.1-Flash, MIT) | weights; reference `inference/` code (dequant, calibration forward) |
| **WestWaters** ([pollard-weights](https://github.com/WestWaters/pollard-weights)) and **bot-lab-21** (DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) | the routed-expert EXL3 pack and its method |
| **turboderp** ([exllamav3](https://github.com/turboderp-org/exllamav3)) | EXL3 format and quantizer (LDLQ) |
| **Kai** | abliteration overlay (`wo_b` layers 10-35, 2026-09-13); the handoff this repo starts from |
| **tonyd2wild** ([DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)) | fleet recipe, bench script `v41bench.py` and prompt set `prompts-v1.json` for apples-to-apples speed |

Add a row the day any idea, kernel or line of code from someone else is adopted.
