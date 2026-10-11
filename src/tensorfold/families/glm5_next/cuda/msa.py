"""TF_GLM_MSA=1: Mia's sparse attention (msa.cu) for a prompt chunk's sparse rows on the FP8 latent cache, in place of
``latent.sparse_onepass``: one CUDA block a row, the rank's heads as one m16 tensor-core tile (up to 16: four Sparks)
or two (up to 32: two and three Sparks), so a tile of keys is loaded and converted once for every head of the row;
the next tile's keys in flight in registers; the codes converted to bf16 with their row's power-of-two scale (exact).
Straight to the output, no partials. A row's bits depend on its query, its token list and the cache only (any
chunking, any rank), but they are not ``sparse_onepass``'s: switching changes the prompt's bits once, so it stays off
(``sparse_onepass``) until the needle checks have run with it.

By Mia's AI Lab (Apache-2.0), adapted from the GLM-5.3 recipe's kernel 2 (its patch 0150), whose design points come
from drowzeys' one-pass prompt attention (github.com/drowzeys/TensorFold, e91312a) and BertholomusAI (Albert Lee)'s
prompt sparse attention (github.com/bertholomus/TensorFold, 8f649f1); no code of theirs."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

HEADS = 32                     # query heads a rank at most (two m16 tiles)
ENABLED = os.environ.get("TF_GLM_MSA", "0") == "1"


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_flash_msa_v2", sources=[str(here / "msa.cpp"), str(here / "msa.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def supported(qa: torch.Tensor, cache: torch.Tensor) -> bool:
    """A CUDA chunk of absorbed queries [R, <= 32, 512] over a 512-wide FP8 latent cache (TF_GLM_KV=fp8, kv8 rows;
    a bf16 cache keeps ``sparse_onepass``, which reads its twice as wide rows faster than this kernel did)."""

    return (qa.is_cuda and qa.dim() == 3 and qa.shape[-1] == 512 and 1 <= qa.shape[1] <= HEADS
            and cache.dtype == torch.uint8 and cache.shape[-1] == 528)


def prompt(qa: torch.Tensor, cache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor, out: torch.Tensor,
           scale: float) -> None:
    """``latent.sparse_onepass``'s call: rows with counts > 0 attend their tokens (ascending, the first counts[r]
    valid) into out[r]; other rows are left alone."""

    if qa.shape[0]:
        _ext().prompt(qa, cache, tokens, counts, out, float(scale))
