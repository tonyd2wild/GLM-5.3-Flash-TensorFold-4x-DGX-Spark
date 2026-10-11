"""EXL3 packs on CUDA, host side: the bytes admission counts for a pack and its extra mapped files."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.numpy import save_file  # noqa: E402

EXL3 = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.0, "head_bits": 6, "codebook": "mul1"}


def test_admission_counts_an_exl3_pack_as_loaded():
    from tensorfold.cuda.geometry import exl3_weights, indexed_weights

    head = {"dtype": "I16", "shape": [320, 15520, 96]}
    assert exl3_weights("lm_head.trellis", head)[0] == 320 * 15520 * 96 * 2 * 7 // 5
    assert exl3_weights("lm_head.suh", {"dtype": "F16", "shape": [5120]})[0] == 5120 * 2
    for skipped in ("model.visual.blocks.0.attn.qkv.weight", "mtp.fc.trellis", "model.language_model.mtp.x.suh"):
        assert exl3_weights(skipped, {"dtype": "BF16", "shape": [64, 64]}) == (0, 0)
    table = {"dtype": "I16", "shape": [1000, 81]}
    name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.trellis"
    assert indexed_weights(1, True)(name, table) == (0, 1000 * 81 * 2)
    assert indexed_weights(1, True)("model.visual.merger.fc.weight", {"dtype": "BF16", "shape": [8, 8]}) == (0, 0)


@pytest.mark.parametrize("suffix", ["shard_0.trellis", "trellis"])
def test_extra_files_add_their_mapped_pages(tmp_path, suffix):
    from tensorfold.cuda.capacity import estimate_weights
    from tensorfold.cuda.geometry import indexed_weights

    shard = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding." + suffix
    save_file({shard: np.zeros((100, 51), dtype=np.int16),
               "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.head_bias":
                   np.zeros((16, 160), dtype=np.float16)}, str(tmp_path / "ngram_embedding.safetensors"))
    got = estimate_weights(tmp_path, indexed_weights(1, True), files=[tmp_path / "ngram_embedding.safetensors"])
    assert got.mapped == 100 * 51 * 2 and got.resident == 16 * 160 * 2
