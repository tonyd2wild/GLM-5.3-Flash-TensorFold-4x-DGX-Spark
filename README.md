# DeepSeek-V4.1-Flash on TensorFold, TP4 on 4× DGX Spark, with our own weights

> **Special credit:** [WestWaters/pollard-weights](https://github.com/WestWaters/pollard-weights) (the method behind
> our EXL3 expert pack) and [tonyd2wild](https://github.com/tonyd2wild) (the 4× Spark serving recipe this fleet runs).
> Full ledger: [CREDITS.md](CREDITS.md).

**Private working repo.** Goal: serve our uncensored `DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated` pack on the
[TensorFold](https://github.com/ashhart/TensorFold) native Zig engine at tensor-parallel 4 across our four DGX Sparks,
using bertholomus's DeepSeek-V4.1 TP4 port. Upstream `main` does not serve DeepSeek-V4.1 yet (tracker
ashhart/TensorFold#559).

Why: the TP4 port reports about 2.5× our vLLM lane's single-stream decode (code 160 / prose 95 tok/s vs about
64) with prefill about even. Those are their numbers on their weights; ours are not measured yet.

## Status

| step | state |
|---|---|
| 1. Inventory: what the engine loads vs what our pack has | **in progress** (2026-10-11) |
| 2. Kernel coverage: which EXL3 bit widths the kit ships | **in progress** |
| 3. Convert our FP8/BF16 dense tensors to EXL3 mul1 (Route A) | not started |
| 4. Rank files (CPU only) | not started; disk is tight on Spark4 and Asusi |
| 5. Boot + gates (needs GLM down: Tony's go) | not started |

Nothing has run on a GPU. GLM-5.3 stays live on `:8000`.

## Layout

| path | what |
|---|---|
| `engine/` | bertholomus/TensorFold, branch `deepseek-v41-zig-tp4` @ `5940208` (Apache-2.0), added as a squashed git subtree. Our engine patches land here as separate commits. Update with `git subtree pull --prefix engine https://github.com/bertholomus/TensorFold.git deepseek-v41-zig-tp4 --squash` |
| `inventory/` | header-only inventory of our pack (`pack_inventory.py`, output in `pollard-abliterated-headers.txt`) |
| `HANDOFF.md` | Kai's handoff (2026-10-11): ground rules, plan, fixes, gates, checkpoints |
| `CREDITS.md` | who we build on |

Recipe and kernel kit: [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10)
(Apache-2.0, tag `v0.1.1`). Not vendored yet; fetched at deploy time.

## Ground rules

- **Our own weights only.** No third-party weight packs.
- **GLM-5.3 on `:8000` is the fleet's live model.** Nothing that stops or crowds it without Tony's go for that window.
- **Never delete under `/var/tmp/models/`** on any node: the packs are hardlinked to each other. New output goes to a new directory.
- **Nothing public** (upstream PRs, issue comments, HF uploads, posts) without Tony's word.

## Our pack today (read from the safetensors headers)

`/var/tmp/models/DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated` on Reddie: 188,245 tensors in 49 hardlinked shards.

| tensors | format today |
|---|---|
| routed experts, 40 layers × 384 × `w1/w3/w2` | **EXL3 mul1** (`trellis` I16, `suh`/`svh` F16, `mul1` I32), 3 or 4 bits per layer |
| `attn.wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b` (40 layers; abliteration overlay on `wo_b` L10-35) | FP8 e4m3 + e8m0 32×32 block scale |
| `ffn.shared_experts.w1/w2/w3` (40 layers) | FP8 |
| `attn.indexer.wq_b` (8 layers), `engram.wkv` (2), `mtp.0.main_proj` | FP8 |
| `head`, `attn.compressor.wkv/wgate`, `attn.indexer.wk`, `indexer.weights_proj`, `ffn.gate.weight` | BF16 |
| `mtp.0-2` blocks: attention + shared experts FP8; 3 × 128 experts | native FP4 (I8-packed + e8m0 scale) |
| norms, `hc_*` (mHC), `attn_sink`, gate bias, embeddings, vision, aligner | BF16 / F32 plain |

## Fleet facts that shape the plan

| node | free disk (2026-10-11) | note |
|---|---|---|
| Reddie | 235 GB | holds every pack |
| Bluey | 559 GB | |
| Spark4 | **47 GB** | cannot hold a ~56+ GB rank file locally |
| Asusi | **37 GB** | same |

Each node has about 20 GiB free memory while GLM serves, so GPU work waits for a window.
