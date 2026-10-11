"""MLX affine projections on CUDA: the stored words, or the four-bit tile layout the shared ``qmm`` kernels read."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch

from tensorfold.cuda.kernels import qmm as shared

from .qmm import lane_matmul


@dataclass
class QLinear:
    weight: torch.Tensor      # original MLX words, or the optimized four-bit tile layout
    scales: torch.Tensor | None
    biases: torch.Tensor | None
    layout: str = "mlx"       # "mlx" as stored, or "tiled" (``tile``)
    rows: int = 0             # N when tiled (the tiled words are padded to 64 columns)
    gs: int = 64              # inputs per quantization group
    bits: int = 4

    @property
    def n(self) -> int:
        return self.rows if self.layout == "tiled" else int(self.weight.shape[0])

    @property
    def k(self) -> int:
        if self.layout == "dense":
            return int(self.weight.shape[1])
        return int(self.weight.shape[1]) * 64 if self.layout == "tiled" else int(self.scales.shape[1]) * self.gs

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases) if t is not None)

    @cached_property
    def fast(self) -> bool:
        return (self.bits, self.gs) == (4, 64) and self.layout != "dense" and all(
            t.dtype == torch.bfloat16 for t in (self.scales, self.biases))


def tile(q: QLinear) -> QLinear:
    if q.layout == "tiled" or not q.fast:
        return q
    p = shared.pack(q.weight, q.scales, q.biases, 64)
    return QLinear(p.weight, p.scales, p.biases, layout="tiled", rows=q.n)


def untile(q: QLinear) -> QLinear:
    """The stored MLX layout again (for the fp32 reference, TP sharding or slicing rows)."""

    if q.layout != "tiled":
        return q
    return QLinear(*shared.unpack(shared.Q4(q.weight, q.scales, q.biases, q.n, q.k, 64)))


def matmul(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """The lane matmul for either layout; both give the same bits."""

    if not q.fast:
        from tensorfold.cuda.kernels.affine import matmul as affine_matmul

        return affine_matmul(x, q)
    if q.layout == "tiled":
        return shared.matmul(x, q, xs)
    return lane_matmul(x, q.weight, q.scales, q.biases, xs=xs)


def matmul_partial(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 sums for a tiled weight, unrounded: a row-parallel rank's share of a projection."""

    if not q.fast:
        from tensorfold.cuda.kernels.affine import matmul as affine_matmul

        return affine_matmul(x, q, f32=True)
    if q.layout != "tiled":
        raise ValueError("matmul_partial takes tiled weights")
    return shared.matmul(x, q, xs, f32=True)


__all__ = ["QLinear", "matmul", "matmul_partial", "tile", "untile"]
