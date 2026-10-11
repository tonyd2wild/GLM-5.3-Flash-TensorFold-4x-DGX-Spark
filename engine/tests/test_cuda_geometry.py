"""Compare startup estimates with actual cache constructors using allocation-only tensors."""

import importlib
import math
import sys
from types import SimpleNamespace, ModuleType

import pytest

from tensorfold.cuda import geometry, capacity


class Allocation:
    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
    def numel(self):
        return math.prod(self.shape)
    def element_size(self):
        return getattr(self.dtype, "itemsize", None) or {"bf16": 2, "fp16": 2, "int16": 2, "int8": 1, "uint8": 1}.get(
            self.dtype, 4)
    def __getitem__(self, index):
        return self
    def __add__(self, other):            # arange(rows)[:, None] + arange(k): the [rows, k] tap table
        return Allocation(self.shape + other.shape, self.dtype, self.device)
    def contiguous(self):
        return self
    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Allocation(shape, self.dtype, self.device)


@pytest.fixture
def allocations(monkeypatch):
    before = set(sys.modules)
    lang = ModuleType("triton.language")
    lang.constexpr = object
    triton = ModuleType("triton")
    triton.language = lang
    triton.jit = lambda fn: fn
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", lang)
    recorded = []
    def allocate(shape, **kw):
        tensor = Allocation(shape, kw.get("dtype", "fp32"), kw.get("device", "cpu"))
        recorded.append(tensor)
        return tensor
    fake = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32", int8="int8", uint8="uint8", int16="int16",
                           int32="int32", int64="int64",
                           zeros=allocate, empty=allocate, full=lambda shape, fill, **kw: allocate(shape, **kw),
                           zeros_like=lambda x: allocate(x.shape, dtype=x.dtype, device=x.device),
                           arange=lambda n, **kw: allocate((n,), **kw),
                           cuda=SimpleNamespace(is_available=lambda: False))
    # Imports use real torch annotations; only the allocation sites are replaced.
    try:
        yield recorded, fake
    finally:
        prefix = "tensorfold.families."
        added = [(name, module) for name, module in list(sys.modules.items())
                 if name not in before and name.startswith(prefix) and ".cuda" in name]
        for name, module in sorted(added, key=lambda item: len(item[0]), reverse=True):
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child, None) is module:
                delattr(parent, child)
            sys.modules.pop(name, None)


def bytes_in(arrays):
    return sum(t.numel() * t.element_size() for t in arrays)


def test_mla_latent_estimate_grows_by_the_cache_and_counts_one_prompt_chunk_scratch():
    """Per token, the latent estimate grows by the latent and indexer caches of every attention layer (the MTP's
    too) plus one fp32 pool score for each prompt-chunk row; the MTP head adds its caches and decode buffers, never a
    second set of the prompt chunk's latent partials (it absorbs through the same prefill buffers)."""

    text = {"hidden_size": 512, "num_attention_heads": 8, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2, "linear_num_heads": 8,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "vocab_size": 1024, "kv_lora_rank": 512,
            "moe_intermediate_size": 512, "num_experts_per_tok": 2}
    rows, heads, lw, index = geometry.PREFILL_ROWS, 4, 512, 128
    a, b = 1 << 18, (1 << 18) + 4096
    const = {}
    for mtp in (0, 1):
        g = geometry.mla_geometry({**text, "num_nextn_predict_layers": mtp}, 2, 8, latent=True)
        count = 2 + mtp
        slope = count * lw * 2 + count * index * 2 * 9 // 4 + rows
        assert g.bytes_at(b) - g.bytes_at(a) == (b - a) * slope
        const[mtp] = g.bytes_at(a) - a * slope
    partials = ((2560 + rows + 511) // 512) * rows * heads * (lw + 2) * 4
    assert 0 < const[1] - const[0] < partials


@pytest.mark.torch
@pytest.mark.parametrize("latent", [True, False], ids=["latent", "per-head"])
@pytest.mark.parametrize("mtp", [False, True])
def test_mla_actual_cache_and_replay_state_are_budgeted(monkeypatch, allocations, mtp, latent):
    arrays, fake = allocations
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    kda = importlib.import_module("tensorfold.families.glm5_next.cuda.kda")
    cache = importlib.import_module("tensorfold.families.glm5_next.cuda.latent")
    monkeypatch.setattr(mod, "torch", fake)
    monkeypatch.setattr(kda, "torch", fake)
    monkeypatch.setattr(cache, "torch", fake)
    monkeypatch.setattr(cache, "ENABLED", latent)
    text = {"hidden_size": 512, "num_attention_heads": 8, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2, "linear_num_heads": 8,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "vocab_size": 1024,
            "moe_intermediate_size": 512, "num_experts_per_tok": 2,
            "num_nextn_predict_layers": int(mtp)}
    cfg = SimpleNamespace(heads=8, lin_heads=8, conv=4, qk_dim=256, v_dim=256, index_dim=128,
                          hidden=512, streams=4, q_lora=512, kv_lora=512, index_heads=32,
                          dense_width=1024, top_k=2, moe_width=512, experts=8, quant="mlx")
    layers = [SimpleNamespace(index=i, kind="kda" if i % 2 == 0 else "dsa",
                              kda=SimpleNamespace(proj=SimpleNamespace(n=3 * 4 * 128 + 256 + 4))) for i in range(4)]
    weights = SimpleNamespace(cfg=cfg, world=2, device="cpu", layers=layers, meta={"long_context": True},
                              mtp=SimpleNamespace() if mtp else None, head=SimpleNamespace(n=512))
    slots = 65536
    attention = importlib.import_module("tensorfold.families.glm5_next.cuda.attention")
    monkeypatch.setattr(attention, "torch", fake)
    mod.Buffers(weights, 64, slots)
    if mtp:
        mod.Buffers(weights, 64, slots)
    mod.Buffers(weights, 2048, slots, prefill=True)     # the prompt chunks' buffers, as ``decode.Engine`` makes them
    mod.State(weights, slots, 64)
    estimated = geometry.mla_geometry(text, 2, 8, latent=latent).bytes_at(slots)
    assert any(slots in t.shape for t in arrays)             # the constructors ran on the fake allocator
    assert bytes_in(arrays) <= estimated


def test_weight_partition_rounding_and_float_casts():
    from tensorfold.families.glm5_next.cuda.split import rule
    transform = geometry.split_weights(rule)
    info = {"dtype": "U32", "shape": [192, 128], "split": False}
    assert transform("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", info) == (128 * 128 * 4, 0)
    assert transform("model.language_model.layers.0.mlp.experts.0.down_proj.weight", info) == (192 * 64 * 4, 0)
    split = {**info, "shape": [96, 128], "split": True}
    assert transform("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", split) == (128 * 128 * 4, 0)
    vector = {"dtype": "BF16", "shape": [64], "split": False}
    assert transform("model.language_model.layers.0.self_attn.A_log", vector) == (32 * 4, 0)
    assert transform("model.visual.weight", info) == (0, 0)


def test_unified_available_memory_uses_reclaimable_host_pages(monkeypatch):
    from pathlib import Path
    fake = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (100 * capacity.GIB, 128 * capacity.GIB)))
    monkeypatch.setattr(Path, "read_text", lambda *a: "MemTotal: 134217728 kB\nMemAvailable: 62914560 kB\n")
    monkeypatch.setattr(capacity, "unified", lambda torch: True)
    # the grant is the reclaimable pool less the floor the host keeps free: a tenth of its 128 GiB of RAM
    assert capacity.available_bytes(fake) == 60 * capacity.GIB - 128 * capacity.GIB // 10
