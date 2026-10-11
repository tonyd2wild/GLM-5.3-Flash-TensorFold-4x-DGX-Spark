"""GLM-5.3-Flash's ModelOpt reader on the host: a tiny mixed-precision pack (NVFP4 routed experts with per-matrix
scalar scales, MXFP8 and FP8-block projections, BF16 elsewhere, a conversion marker, a shard's stale duplicate) read
for each of four ranks, from the full checkpoint and from rank folders: every rank's share and its 8-bit linears and
NVFP4 experts against the dequantized whole."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.torch

P = "model.language_model."
WORLD = 4
D, NI, HEADS, QK, LH, Q_LORA = 256, 512, 4, 128, 4, 256       # every family whole at TP4 in its unit
E = 2                                                           # routed experts
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

TEXT = {"hidden_size": D, "num_attention_heads": HEADS, "linear_attn_config": {"num_heads": LH, "head_dim": 128},
        "intermediate_size": 512, "moe_intermediate_size": NI, "n_shared_experts": 1, "vocab_size": 256,
        "layer_types": ["linear_attention", "full_attention"], "num_hidden_layers": 2}


def _raw(t):
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().copy()


def _nvfp4(n: int, k: int, g):
    words = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    scales = (torch.rand((n, k // 16), generator=g) * 4 + 0.25).to(torch.float8_e4m3fn)
    return words, scales, torch.tensor(0.03125 * (1 + torch.rand((), generator=g).item()), dtype=torch.float32)


def _nvfp4_dense(words, scales, glob):
    codes = torch.stack([words & 0xF, words >> 4], dim=-1).reshape(words.shape[0], -1).long()
    mags = torch.tensor(E2M1 + tuple(-v for v in E2M1))
    return mags[codes] * scales.float().repeat_interleave(16, dim=1) * float(glob)


def _mx8(n: int, k: int, g):
    w = (torch.randn((n, k), generator=g) * 2).to(torch.float8_e4m3fn)
    e = torch.randint(118, 132, (n, k // 32), dtype=torch.uint8, generator=g)
    return w, e


def _mx8_dense(w, e):
    return (w.float() * torch.exp2(e.float() - 127).repeat_interleave(32, dim=1)).to(torch.bfloat16)


def _fp8b(n: int, k: int, g):
    w = (torch.randn((n, k), generator=g) * 2).to(torch.float8_e4m3fn)
    s = torch.rand((-(-n // 128), k // 128), generator=g) * 0.01 + 0.001
    return w, s


def _fp8b_dense(w, s):
    n, k = w.shape
    full = s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return (w.float() * full).to(torch.bfloat16)


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    """The checkpoint folder and its tensors (name -> (dtype, tensor)) as one rank-less whole."""

    from tensorfold.families.glm5_next.cuda.split import write

    g = torch.Generator().manual_seed(7)
    t: dict[str, tuple[str, torch.Tensor]] = {}
    for e in range(E):
        for proj, (n, k) in (("gate_proj", (NI, D)), ("up_proj", (NI, D)), ("down_proj", (D, NI))):
            words, scales, glob = _nvfp4(n, k, g)
            base = f"{P}layers.1.mlp.experts.{e}.{proj}."
            t[base + "weight"] = ("U8", words)
            t[base + "weight_scale"] = ("F8_E4M3", scales)
            t[base + "weight_scale_2"] = ("F32", glob)
    w, e8 = _mx8(D, LH * 128, g)                                    # a KDA layer's o_proj: MXFP8, column-split
    t[P + "layers.0.self_attn.o_proj.weight"] = ("F8_E4M3", w)
    t[P + "layers.0.self_attn.o_proj.weight_scale"] = ("U8", e8)
    for name in ("q_a_proj", "kv_a_proj_with_mqa"):                  # replicated FP8-block rows of one input
        w, s = _fp8b(Q_LORA if name == "q_a_proj" else 200, D, g)
        t[P + f"layers.1.self_attn.{name}.weight"] = ("F8_E4M3", w)
        t[P + f"layers.1.self_attn.{name}.weight_scale_inv"] = ("F32", s)
    w, s = _fp8b(HEADS * QK, Q_LORA, g)                             # q_b: FP8 blocks split by heads
    t[P + "layers.1.self_attn.q_b_proj.weight"] = ("F8_E4M3", w)
    t[P + "layers.1.self_attn.q_b_proj.weight_scale_inv"] = ("F32", s)
    for name in ("indexer.wk", "indexer.weights_proj"):              # replicated BF16
        t[P + f"layers.1.self_attn.{name}.weight"] = ("BF16", torch.randn((128 if "wk" in name else 32, D),
                                                                             generator=g).to(torch.bfloat16))
    t["qmix.requires_patch"] = ("U8", torch.ones((1,), dtype=torch.uint8))
    a = {k: v for i, (k, v) in enumerate(t.items()) if i % 2 == 0}
    b = {k: v for k, v in t.items() if k not in a}
    stale = P + "layers.1.self_attn.q_b_proj.weight"                # file B keeps an older copy of a tensor A owns
    if stale not in a:
        a[stale], b = t[stale], {k: v for k, v in b.items() if k != stale}
    b_with_stale = dict(b)
    b_with_stale[stale] = ("F8_E4M3", torch.zeros_like(t[stale][1]))
    root = tmp_path_factory.mktemp("modelopt_pack")
    files = {"model-00001-of-00002.safetensors": a, "model-00002-of-00002.safetensors": b_with_stale}
    for fname, tensors in files.items():
        write(str(root / fname), [(k, dt, list(x.shape), _raw(x)) for k, (dt, x) in tensors.items()], None)
    index = {k: "model-00001-of-00002.safetensors" for k in a} | {k: "model-00002-of-00002.safetensors" for k in b}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (root / "config.json").write_text(json.dumps({
        "text_config": TEXT, "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}}))
    return root, t


def _readers(root, tmp_path, rank: int):
    """Rank ``rank``'s reader of the full checkpoint and of a rank folder split from it."""

    from tensorfold.families.glm5_next.cuda import split

    out = tmp_path / f"rank{rank}"
    layout = split.layout_for(root, rank, WORLD)
    for src in sorted(root.glob("model-*.safetensors")):
        split.split_file(src, out, rank, WORLD, layout)
    return split.RankReader(root, rank, WORLD), split.RankReader(out, rank, WORLD)


def test_split_rules_scalars_and_dtypes():
    from tensorfold.families.glm5_next.cuda import split, tp

    assert split.rule("qmix.requires_patch") == "drop"
    assert split.rule("model.visual.blocks.0.attn.qkv.weight") == "drop"
    assert split.rule(P + "layers.3.mlp.experts.0.down_proj.weight_scale_2") == "col"
    assert split.torch_dtype("F8_E4M3") is torch.float8_e4m3fn and split.DTYPE_BYTES["F8_E8M0"] == 1
    raw = np.frombuffer(np.float32(0.5).tobytes(), dtype=np.uint8)
    for kind in ("row", "col"):
        data, shape = split.split_bytes(raw, [], 4, kind, 3)
        assert shape == [] and data.tobytes() == raw.tobytes()
        dev, dshape = split.split_device(torch.from_numpy(raw.copy()), [], 4, kind, 3)
        assert dshape == [] and dev.numpy().tobytes() == raw.tobytes()
    lay = tp.Layout.from_text(TEXT, 2, WORLD)
    assert split.rank_cut(lay, P + "layers.1.mlp.experts.0.gate_proj.weight_scale_2", [], "row") is None
    assert split.rank_cut(lay, P + "layers.1.mlp.experts.0.down_proj.weight_scale", [D, NI // 16], "col") == (16, 24)


def test_headers_count_the_indexed_copy_and_the_estimate_takes_scalars(pack):
    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.glm5_next.cuda import split, tp

    root, t = pack
    found = headers(root)                                     # no "duplicate checkpoint tensor" for the stale copy
    assert set(found) == set(t)
    lay = tp.Layout.from_text(TEXT, 1, WORLD)

    def cut(name, shape, kind):
        got = split.rank_cut(lay, name, shape, kind)
        return got if got is not None else (0, shape[{"row": 0, "col": -1, "dim1": 1}[kind]] // WORLD)

    est = split_weights(split.rule, WORLD, cut)
    sizes = {name: est(name, info)[0] for name, info in found.items()}
    assert sizes["qmix.requires_patch"] == 0
    assert sizes[P + "layers.1.mlp.experts.0.gate_proj.weight_scale_2"] == 4          # whole on every rank
    assert sizes[P + "layers.1.mlp.experts.0.down_proj.weight"] == D * NI // 2 // WORLD
    assert sizes[P + "layers.1.self_attn.q_b_proj.weight_scale_inv"] == 4 * (HEADS * QK // 128 // WORLD) * 2


@pytest.mark.parametrize("rank", range(WORLD))
def test_rank_shares_and_linears(pack, tmp_path, rank):
    from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear, Mx8Linear
    from tensorfold.families.glm5_next.cuda import modelopt

    root, t = pack
    full, folder = _readers(root, tmp_path, rank)
    try:
        q_b = P + "layers.1.self_attn.q_b_proj"
        assert t[q_b + ".weight"][1].view(torch.uint8).any()       # unlike the stale copy (zeros)
        for rd in (full, folder):
            assert "qmix.requires_patch" not in rd.index or rd is full    # a rank folder drops the marker
            # the stale copy never wins: the rank's rows of the indexed one
            rows = slice(rank * HEADS * QK // WORLD, (rank + 1) * HEADS * QK // WORLD)
            assert torch.equal(rd.get(q_b + ".weight").view(torch.uint8), t[q_b + ".weight"][1][rows].view(torch.uint8))
            # FP8 blocks split by heads: the rank's block rows, as one Fp8BlockLinear
            lin = modelopt.linears(rd, [q_b], torch.device("cpu"))
            assert isinstance(lin, Fp8BlockLinear) and (lin.n, lin.k) == (HEADS * QK // WORLD, Q_LORA)
            want = _fp8b_dense(*(t[q_b + x][1] for x in (".weight", ".weight_scale_inv")))[rows]
            assert torch.equal(modelopt.dense(rd, q_b, torch.device("cpu")), want)
            s_rows = slice(rank, rank + 1)
            cols = Fp8BlockLinear.column_scales(t[q_b + ".weight_scale_inv"][1][s_rows], lin.n, lin.k)
            assert torch.equal(lin.scale_rows(), cols)
            # replicated FP8-block rows of one input stacked: the parts' rows and scales in order
            names = [P + "layers.1.self_attn.q_a_proj", P + "layers.1.self_attn.kv_a_proj_with_mqa"]
            both = modelopt.linears(rd, names, torch.device("cpu"))
            assert isinstance(both, Fp8BlockLinear) and both.n == Q_LORA + 200
            assert torch.equal(both.scale_rows(), torch.cat([Fp8BlockLinear.column_scales(
                t[n + ".weight_scale_inv"][1], t[n + ".weight"][1].shape[0], D) for n in names]))
            # MXFP8 o_proj split by KDA heads along its inputs
            o = P + "layers.0.self_attn.o_proj"
            mx = modelopt.linears(rd, [o], torch.device("cpu"))
            c0, c1 = rank * LH * 128 // WORLD, (rank + 1) * LH * 128 // WORLD
            assert isinstance(mx, Mx8Linear) and (mx.n, mx.k, mx.npad) == (D, LH * 128 // WORLD, D)
            assert torch.equal(mx.w8_rows().view(torch.uint8), t[o + ".weight"][1][:, c0:c1].view(torch.uint8))
            assert torch.equal(mx.scale_rows(), t[o + ".weight_scale"][1][:, c0 // 32:c1 // 32])
            assert torch.equal(modelopt.dense(rd, o, torch.device("cpu")),
                               _mx8_dense(t[o + ".weight"][1], t[o + ".weight_scale"][1])[:, c0:c1])
            # BF16 parts: no 8-bit linear; ``projection`` reads them once as bf16
            ix = [P + "layers.1.self_attn.indexer.wk", P + "layers.1.self_attn.indexer.weights_proj"]
            assert modelopt.linears(rd, ix, torch.device("cpu")) is None
            got = modelopt.projection(rd, ix, torch.device("cpu"))
            assert [x.dtype for x in got] == [torch.bfloat16] * 2
            assert all(torch.equal(x, t[n + ".weight"][1]) for x, n in zip(got, ix))
            # parts stored otherwise (FP8 blocks beside BF16): each as bf16, the 8-bit one dequantized
            mixed = modelopt.projection(rd, [q_b, ix[0]], torch.device("cpu"))
            assert isinstance(mixed, list) and torch.equal(mixed[0], want)
    finally:
        full.close()
        folder.close()


@pytest.mark.parametrize("rank", range(WORLD))
def test_rank_nvfp4_experts(pack, tmp_path, rank):
    from tensorfold.cuda.nvfp4 import experts as nvx
    from tensorfold.families.glm5_next.cuda import modelopt

    root, t = pack
    full, folder = _readers(root, tmp_path, rank)
    prefix = P + "layers.1.mlp."
    w0, w1 = rank * NI // WORLD, (rank + 1) * NI // WORLD
    try:
        for rd in (full, folder):
            names = modelopt.expert_parts(rd, prefix, E)
            assert len(names) == 3 * E * 3 and names[0] == prefix + "experts.0.gate_proj.weight"
            rd.prefetch(names, None)                               # the loader's read-ahead (host here)
            ex = modelopt.experts(rd, prefix, E, torch.device("cpu"), 7.0)
            assert isinstance(ex, nvx.Experts4) and (ex.width, ex.dims, ex.count, ex.limit) == (NI // WORLD, D, E, 7.0)
            assert tuple(ex.up.shape) == (E, NI // WORLD // 32, D // 32, 2, nvx.WORDS)
            assert tuple(ex.down.shape) == (E, D // 32, NI // WORLD // 32, 1, nvx.WORDS)
            for e in range(E):
                for which, proj, sl in (("gate", "gate_proj", (slice(w0, w1), slice(None))),
                                        ("up", "up_proj", (slice(w0, w1), slice(None))),
                                        ("down", "down_proj", (slice(None), slice(w0, w1)))):
                    base = f"{prefix}experts.{e}.{proj}."
                    ref = _nvfp4_dense(*(t[base + x][1] for x in ("weight", "weight_scale", "weight_scale_2")))
                    assert torch.equal(nvx.dense(ex, e, which), ref[sl]), (rank, e, which)
    finally:
        full.close()
        folder.close()


def test_bf16_experts_take_nvfp4_at_load():
    """The MTP layer's BF16 experts: ModelOpt's NVFP4 recipe at load (drafts only), close to the BF16 weights."""

    from tensorfold.cuda.nvfp4 import experts as nvx
    from tensorfold.families.glm5_next.cuda import modelopt

    g = torch.Generator().manual_seed(3)
    t = {}
    for e in range(3):
        for proj, shape in (("gate_proj", (64, 128)), ("up_proj", (64, 128)), ("down_proj", (128, 64))):
            t[f"m.experts.{e}.{proj}.weight"] = torch.randn(shape, generator=g).to(torch.bfloat16)

    class Reader:
        index = dict.fromkeys(t)

        @staticmethod
        def get(name):
            return t[name]

    assert modelopt.expert_parts(Reader, "m.", 3) == [f"m.experts.{e}.{p}.weight" for p in
                                                        ("gate_proj", "up_proj", "down_proj") for e in range(3)]
    ex = modelopt.experts(Reader, "m.", 3, torch.device("cpu"), 0.0)
    assert (ex.width, ex.dims, ex.count) == (64, 128, 3)
    for e in range(3):
        for which in ("gate", "up", "down"):
            ref = t[f"m.experts.{e}.{which}_proj.weight"].float()
            err = (nvx.dense(ex, e, which) - ref).norm() / ref.norm()
            assert err < 0.15, (e, which, float(err))


def test_dense_refuses_what_it_cannot_read():
    from tensorfold.families.glm5_next.cuda import modelopt

    t = {"a.weight": torch.zeros((4, 64), dtype=torch.float8_e4m3fn), "b.weight": torch.zeros((4, 64))}

    class Reader:
        index = dict.fromkeys(t)

        @staticmethod
        def get(name):
            return t[name]

    with pytest.raises(ValueError, match="without weight_scale_inv or weight_scale"):
        modelopt.dense(Reader, "a", torch.device("cpu"))
    with pytest.raises(ValueError, match="is not part of this checkpoint format"):
        modelopt.dense(Reader, "b", torch.device("cpu"))
