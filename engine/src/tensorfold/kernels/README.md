# Kernel layout

| Family | Active Metal kernel package |
| --- | --- |
| GLM-5.3-Flash | `glm/flash/v1/` |
| Nemotron 3.5 Lightning | `nemotron/lightning/v1/` |
| Gemma 4 | `gemma/v1/` |

The version names the implementation, not the model release. Each family imports its active package;
incompatible versions can occupy separate directories. CUDA kernels live in the family's `cuda/` package.

Snapshot fingerprints hash family code, active kernels and declared shared dependencies.
`qwen/dense/v1/` and `qwen/flash_next/v1/` are no family's active package; they hold kernels other families share. GLM declares `qwen/flash_next/v1/prefill_mm` as a shared dependency. Nemotron declares dense-Qwen attention as a shared dependency. Gemma declares dense-Qwen's lane matmul and Nemotron's row matvecs. The snapshot key also includes the
prefill plan and the MLX, mlx-lm and TensorFold versions.

See [the family guide](../../../docs/recipes/adding-a-family.md) for arithmetic, state and verification
requirements. A multi-row kernel must match its own serial row before it can verify drafts.
