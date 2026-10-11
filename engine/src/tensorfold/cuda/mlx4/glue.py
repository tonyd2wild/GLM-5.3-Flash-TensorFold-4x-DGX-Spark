"""Each kernel reads only its row and writes fp32 group sums from its stored bf16 values, preserving row bits independently of the window."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm(X, R, W, H, Y, XS, eps, D: tl.constexpr, BLOCK: tl.constexpr, HAS_R: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    if HAS_R:
        r = tl.load(R + row * D + offs, mask=ok, other=0.0).to(tl.float32)
        h = (x + r).to(tl.bfloat16)
        tl.store(H + row * D + offs, h, mask=ok)
        x = h.to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    inv = 1.0 / tl.sqrt(ss / D + eps)
    w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    y = (x * inv * w).to(tl.bfloat16)
    tl.store(Y + row * D + offs, y, mask=ok)
    yg = tl.reshape(y.to(tl.float32), (BLOCK // 64, 64))
    g = tl.arange(0, BLOCK // 64)
    tl.store(XS + row * (D // 64) + g, tl.sum(yg, axis=1), mask=g < D // 64)


def add_rmsnorm(x: torch.Tensor, r: torch.Tensor | None, w: torch.Tensor, eps: float):
    """h = x + r (bf16), y = rmsnorm(h) * w (bf16), xs = y's 64-group sums. Returns (h, y, xs)."""

    rows, d = x.shape
    h = torch.empty_like(x) if r is not None else x
    y = torch.empty_like(x)
    xs = torch.empty((rows, d // 64), dtype=torch.float32, device=x.device)
    _add_rmsnorm[(rows,)](x, r if r is not None else x, w, h, y, xs, eps, D=d,
                          BLOCK=triton.next_power_of_2(d), HAS_R=r is not None, num_warps=8)
    return h, y, xs


@triton.jit
def _embed(IDS, Wt, S, B, OUT, D: tl.constexpr):
    row = tl.program_id(0)
    g = tl.program_id(1)
    tok = tl.load(IDS + row).to(tl.int64)
    words = tl.load(Wt + tok * (D // 8) + g * 8 + tl.arange(0, 8))
    q = (words[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF
    q = tl.reshape(q, (64,)).to(tl.float32)
    s = tl.load(S + tok * (D // 64) + g).to(tl.float32)
    b = tl.load(B + tok * (D // 64) + g).to(tl.float32)
    tl.store(OUT + row * D + g * 64 + tl.arange(0, 64), (q * s + b).to(tl.bfloat16))


def embed(ids: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, d: int) -> torch.Tensor:
    W = ids.shape[0]
    out = torch.empty((W, d), dtype=torch.bfloat16, device=ids.device)
    _embed[(W, d // 64)](ids, weight, scales, biases, out, D=d, num_warps=1)
    return out


__all__ = ["add_rmsnorm", "embed"]
