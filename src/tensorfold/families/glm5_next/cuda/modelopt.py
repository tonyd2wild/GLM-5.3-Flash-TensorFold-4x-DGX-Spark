"""ModelOpt mixed-precision GLM-5.3-Flash checkpoints: NVFP4 routed experts as stored, every other projection to BF16.

The pack this reads (``quant_method: modelopt``, ``quant_algo: MIXED_PRECISION``) stores
- routed experts as NVFP4: ``weight`` uint8 (two e2m1 codes a byte, low nibble first), ``weight_scale`` e4m3 per 16
  inputs and ``weight_scale_2`` one fp32 per matrix, which ``tensorfold.cuda.nvfp4.experts.make`` takes as stored;
- linear-attention projections as MXFP8: e4m3 ``weight`` with a uint8 e8m0 exponent per 32 inputs (``weight_scale``);
- sparse-attention projections and shared experts as FP8 with fp32 ``weight_scale_inv`` per 128x128 block;
- the rest (and the MTP layer) as BF16.
Dense projections become BF16 at load (about 3 GB a rank at TP4); the MTP layer's BF16 experts take ModelOpt's NVFP4
recipe at load, since they only propose drafts.
"""

from __future__ import annotations

import torch

from tensorfold.cuda.nvfp4 import experts as nvx

FP8_BLOCK = 128           # rows and inputs a weight_scale_inv entry covers
MX_GROUP = 32             # inputs an MXFP8 exponent covers
QUANT_CHUNK = 8           # BF16 experts quantized at a time (bounds the fp32 working copy)


def has(rd, name: str) -> bool:
    return name in rd.index


def dense(rd, name: str, device: torch.device) -> torch.Tensor:
    """``name`` (without ``.weight``) -> this rank's [N, K] bf16 on ``device``."""

    w = rd.get(name + ".weight").to(device)
    if w.dtype == torch.bfloat16:
        return w
    if w.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name}: a {w.dtype} dense weight is not part of this checkpoint format")
    n, k = w.shape
    if has(rd, name + ".weight_scale_inv"):
        s = rd.get(name + ".weight_scale_inv").to(device, torch.float32)
        s = s.repeat_interleave(FP8_BLOCK, 0)[:n].repeat_interleave(FP8_BLOCK, 1)[:, :k]
        return (w.to(torch.float32) * s).to(torch.bfloat16)
    if has(rd, name + ".weight_scale"):
        e = rd.get(name + ".weight_scale").to(device)
        if e.dtype != torch.uint8 or tuple(e.shape) != (n, k // MX_GROUP):
            raise ValueError(f"{name}: MXFP8 scales {tuple(e.shape)} {e.dtype} do not fit a [{n}, {k}] weight")
        scale = torch.exp2(e.to(torch.float32) - 127.0).repeat_interleave(MX_GROUP, 1)
        return (w.to(torch.float32) * scale).to(torch.bfloat16)
    raise ValueError(f"{name}: an FP8 weight without weight_scale_inv or weight_scale")


def expert_parts(rd, prefix: str, count: int) -> list[str]:
    """The tensors ``experts`` reads for ``prefix`` (``...layers.N.mlp.``), in order, for read-ahead."""

    names = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for e in range(count):
            base = f"{prefix}experts.{e}.{proj}."
            names += [base + x for x in ("weight", "weight_scale", "weight_scale_2") if has(rd, base + x)]
    return names


def experts(rd, prefix: str, count: int, device: torch.device, limit: float) -> nvx.Experts4:
    """``prefix``'s routed experts as one grouped NVFP4 layer (stored NVFP4, or BF16 quantized at load)."""

    parts = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        names = [f"{prefix}experts.{e}.{proj}" for e in range(count)]
        if has(rd, names[0] + ".weight_scale_2"):
            words = torch.stack([rd.get(n + ".weight").to(device) for n in names])
            scales = torch.stack([rd.get(n + ".weight_scale").to(device).view(torch.uint8) for n in names])
            glob = torch.stack([rd.get(n + ".weight_scale_2").to(torch.float32).reshape(()) for n in names]).to(device)
        else:
            ws, ss, gs = [], [], []
            for e0 in range(0, count, QUANT_CHUNK):
                chunk = torch.stack([rd.get(n + ".weight").to(device) for n in names[e0:e0 + QUANT_CHUNK]])
                w, s, g = nvx.quantize(chunk)
                ws.append(w); ss.append(s); gs.append(g)
                del chunk
            words, scales, glob = torch.cat(ws), torch.cat(ss), torch.cat(gs)
        parts[proj] = (words.contiguous(), scales.contiguous(), glob.contiguous())
    ex = nvx.make(parts["gate_proj"], parts["up_proj"], parts["down_proj"], limit=limit)
    del parts
    return ex


# -- 8-bit projections as stored ---------------------------------------------------------------------------------

def _stored(rd, name: str, device: torch.device):
    """``name``'s weight and its 8-bit layout: ("mx8", w, e8m0 scales), ("fp8b", w, scale_inv) or ("b16", None, None)."""

    w = rd.get(name + ".weight").to(device)
    if w.dtype == torch.float8_e4m3fn and has(rd, name + ".weight_scale_inv"):
        return "fp8b", w, rd.get(name + ".weight_scale_inv").to(device, torch.float32)
    if w.dtype == torch.float8_e4m3fn and has(rd, name + ".weight_scale"):
        return "mx8", w, rd.get(name + ".weight_scale").to(device)
    return "b16", None, None


def linears(rd, names: list[str], device: torch.device):
    """``names`` (one input) as one projection on the 8-bit kernels when all are stored alike, else None (BF16)."""

    from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear, Mx8Linear

    parts = [_stored(rd, n, device) for n in names]
    kinds = {p[0] for p in parts}
    if kinds == {"mx8"}:
        return Mx8Linear.from_checkpoint(torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]))
    if kinds == {"fp8b"}:
        w = torch.cat([p[1] for p in parts])
        cols = torch.cat([Fp8BlockLinear.column_scales(p[2], p[1].shape[0], p[1].shape[1]) for p in parts])
        return Fp8BlockLinear.from_rows(w, cols)
    return None
