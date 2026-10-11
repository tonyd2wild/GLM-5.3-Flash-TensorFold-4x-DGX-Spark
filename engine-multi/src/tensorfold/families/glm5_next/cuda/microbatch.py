"""TF_GLM_PROMPT_MICROBATCH=1: a row-split prompt chunk in two halves, one's exchanges under the other's compute.

The row split (``hcsplit``) exchanges, at each of a layer's two sites, every rank's fp32 partials of the other ranks'
rows, then glues its own rows and swaps the glued (normed) rows. With the overlap (TF_GLM_PREFILL_OVERLAP) the
exchanges run on a side stream, but the main stream soon waits for them: the glue needs the partials, the next block
the glued rows, and little else is left to compute. Past two ranks those exchanges are most of a chunk's time beside
the compute (four Sparks on a ring: ~24 MiB of partials a rank and site).

Here a chunk of R rows at position p is two halves, A = rows [0, h) at p and B = rows [h, R) at p + h (h a multiple
of 64, KDA's grid, and of the world size, so A has no pad row), each row-split by its own ``HcSplit`` over row views
of the prompt buffers (``view``: the row-indexed tensors' rows, its own region of the partial staging; the main
stream's scratch shared), both on one side stream. Per layer the main stream runs

    A's KDA / DSA block, B's, A's MoE (or MLP), B's

and as soon as a half's block has filled its partials, its chain (the partials' exchange, then the glue on its own
rows and the glued rows' swap, ``HcSplit.glue_side``) is queued on the side stream, so it runs while the main stream
computes the other half: A's after-attention chain under B's attention, B's under A's MoE, A's layer-end chain under
B's MoE, B's under A's next attention. The main stream waits for a half's chain (``join``) only right before that
half's next block.

Bits: half A is a chunk of h rows at p and half B one of R - h rows at p + h, layer by layer; a row's arithmetic does
not depend on the chunking (the kernels keep rows apart, partials are summed rank 0 first, a row's selection depends
on its query and the keys at or before it, KDA's chunked recurrence runs from the state at the half's first row, on
its 64-row grid), so every bit is the whole chunk's. The order keeps every dependency of two chunks run one after the
other: B's KDA continues from A's state of the layer, B's attention reads A's latents and index keys of the layer
(A's block came first on the main stream); nothing of A reads B. The side stream touches only the half's own rows and
its own region of the partial staging.

With TF_GLM_PROMPT_OWN_FRONT each half's DSA block runs its own-row front (``own_front``) on the main stream as before,
its shares on the side stream. The glue's fronts (TF_GLM_PREFILL_OVERLAP=2) are not run during the halves' glue: the
other half's compute covers the exchanges instead. Without a side stream (the CPU tests) the chains run in place, in
the same order: the same bits, no overlap.

Halves of fewer than TF_GLM_PROMPT_MICROBATCH_MIN_ROWS rows (default 2,048) run the chunk whole: the routed experts
decode each expert's weights once a call, so two calls of half the rows cost more (GB10, real routing: 1,024-row calls
take 23-73% longer a row than 2,048-row ones at every width a rank holds; 2,048-row ones 4-31% longer than 4,096-row
ones). Meant with TF_GLM_PREFILL_ROWS=4096: then each half pays today's 2,048-row cost.

Not for several streams' prompts in one chunk (``multi_prefill``), nor without the latent cache. By Mia's AI Lab
(Apache-2.0), ported from the GLM-5.3 recipe's patch 0148 to Flash's KDA layers and hyper-connection glue. Design
credit: drowzeys (github.com/drowzeys/TensorFold, branch glm53-tp4-2026-10-05, ``compute_prompt_sp`` with
TF_GLM53_PROMPT_OVERLAP) authored the two micro-batches whose exchange chains run on a communication stream under the
other half's compute, which this follows; no code of theirs."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import torch

from . import glue, latent, prof

# the prompt buffers' row-indexed tensors (rows first): a half's view is rows [lo, lo + n) of each
ROWS = ("ids", "hin", "x", "normed", "xs", "post", "comb", "hcpart", "ka", "kg", "xs_fa", "xs_ga", "kxs", "dp", "qr",
        "xs_qr", "lat", "xs_lat", "q", "kn", "vn", "xs_ao", "ikr", "igr", "qi", "gu", "act", "xs_act", "mlog", "pick",
        "wts", "ey", "sgu", "sact", "sxs", "sy", "part", "hidden", "fnormed", "fxs", "me", "mcat", "mxs", "mx")
GRID = 64                 # KDA's chunked recurrence keeps prompt rows on a 64-row grid


def cut(R: int, world: int) -> int:
    """Half A's rows of a chunk of R: about half, a multiple of 64 and of the world size."""

    unit = GRID * world // __import__("math").gcd(GRID, world)
    return max(unit, (R // 2) // unit * unit)


def applies(w, st, b, mixer, host_pos, R: int) -> bool:
    """Whether a row-split prompt chunk of R rows runs as two halves: TF_GLM_PROMPT_MICROBATCH, one stream's prompt on
    ``st`` (no ``mixer``), its position known on the host, the latent cache, and both halves of at least the minimum
    rows (and of what the split takes)."""

    sp = b.split
    if not (sp is not None and sp.settings.microbatch and mixer is None and st is not None and host_pos is not None
            and b.prefill and latent.ENABLED):
        return False
    h = cut(R, sp.world)
    least = max(sp.settings.micro_min, sp.settings.min_rows)
    return h >= least and R - h >= least


def view(b, lo: int, n: int, world: int, gath_lo: int):
    """The prompt buffers' rows [lo, lo + n) (pad rows past them included up to ``n``) as a Buffers-like object for a
    half: every row-indexed tensor's rows, the half's own region of the partial staging from ``gath_lo``; the scratch
    the main stream uses one call at a time (attention partials, the experts', the matmuls') shared."""

    v = copy.copy(b)
    v.__dict__.pop("own_lists", None)
    for name in ROWS:
        t = getattr(b, name, None)
        if isinstance(t, torch.Tensor):
            setattr(v, name, t[lo:lo + n])
    slots = b.top_k + 1
    v.eact = b.eact[lo * slots:(lo + n) * slots]
    if hasattr(b, "kproj"):
        v.kproj = b.kproj[:, lo:lo + n]
        k = b.kscratch
        v.kscratch = SimpleNamespace(out=k.out[lo:lo + n], k=k.k[lo:lo + n], v=k.v[lo:lo + n], g=k.g[lo:lo + n],
                                     b=k.b[lo:lo + n])
    if b.lat_s is not None:
        v.lat_s = copy.copy(b.lat_s)
        v.lat_s.qa, v.lat_s.ol = b.lat_s.qa[lo:lo + n], b.lat_s.ol[lo:lo + n]
    v.taps = [t[lo:lo + n] for t in b.taps]
    d = b.part.shape[1]
    v.gath = b.gath[gath_lo:gath_lo + world * n * d]
    v.rows = v.alloc_rows = n
    v.split = None
    return v


class Halves:
    """A chunk's two halves: their buffer views, states and splits (kept on the buffers for chunks of one shape)."""

    def __init__(self, w, b, R: int) -> None:
        sp = b.split
        world = sp.world
        h = cut(R, world)
        nb = sp.padded(R - h)                 # B's rows with its pad rows (the chunk's last rows of the buffers)
        if h + nb > b.alloc_rows:
            raise ValueError(f"microbatch: halves of {h} and {nb} rows past the buffers' {b.alloc_rows}")
        self.h, self.R = h, R
        settings = dataclasses.replace(sp.settings, pieces=1, microbatch=False)
        d = b.part.shape[1]
        self.a = view(b, 0, h, world, 0)
        self.b = view(b, h, nb, world, world * h * d)
        from .hcsplit import HcSplit

        self.a.split = HcSplit(w, self.a, settings, stream=sp.stream)
        self.b.split = HcSplit(w, self.b, settings, stream=sp.stream)


def _halves(w, b, R: int) -> Halves:
    cache = b.__dict__.setdefault("micro_halves", {})
    key = (R, id(b.split))
    hv = cache.get(key)
    if hv is None:
        if len(cache) >= 4:                   # a few chunk shapes: the full chunk and a prompt's tails
            cache.pop(next(iter(cache)))
        hv = cache[key] = Halves(w, b, R)
    return hv


def _state(st, at: int):
    """``st`` from ``at`` rows on: the same caches and KDA state (shared objects), its position ``at`` rows later."""

    s = copy.copy(st)
    s.pos = st.pos + at
    s.pos_dev = st.pos_dev + at
    return s


def layer_steps(w, st, b, R: int, nch: int | None, host_pos: int):
    """``forward.split_layer_steps`` for a chunk in two halves: every layer, yielding after each; b.hidden (and the
    taps) whole at the end, as the unhalved loop leaves them."""

    from . import own_front
    from .forward import _ffn, _mixer, chunks_for

    c = w.cfg
    layers = w.layers
    hv = _halves(w, b, R)
    halves = ((hv.a, st, hv.h, host_pos), (hv.b, _state(st, hv.h), R - hv.h, host_pos + hv.h))
    own = own_front.applies(w, st, b, None, host_pos)
    for half, _, n, _ in halves:
        half.split.begin(n)
    b.split.micro = [hv.a.split, hv.b.split]
    try:
        # the first hc_pre on every row, as unsplit (B's pad rows: zero streams, as its split's begin left them)
        Rp = hv.h + hv.b.split.Rp
        first, h0 = layers[0], layers[0].attn_hc
        glue.hc_pre(b.x[:Rp], h0.fn, h0.base, h0.scale, first.in_norm, b.normed[:Rp], b.xs[:Rp], b.post[:Rp],
                    b.comb[:Rp], b.hcpart[:Rp], c.eps, c.hc_eps, c.hc_iters, prompt=True)
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            for half, hs, n, pos in halves:
                half.split.join()
                if own and own_front.takes(layer):
                    with prof.timed("dsa (total)"):
                        own_front.dsa_block(layer, w, hs, half, n, chunks_for(hs, n), pos)
                else:
                    _mixer(layer, w, hs, half, n, chunks_for(hs, n), pos, None)
                with prof.timed("hc"):
                    half.split.glue_side(layer.ffn_hc, layer.post_norm)
            for half, hs, n, pos in halves:
                half.split.join()
                _ffn(layer, w, half, n)
                with prof.timed("hc (layer end)"):
                    half.split.glue_side(nxt.attn_hc if nxt is not None else None,
                                         nxt.in_norm if nxt is not None else None,
                                         taps=tuple(b.tap_at.get(layer.index, ())), final=nxt is None,
                                         share=not (own and nxt is not None and own_front.takes(nxt)))
            yield layer.index
    finally:
        for half, _, _, _ in halves:
            half.split.finish()
        b.split.micro = []
        b.split.finish()                      # compute_steps re-activated the chunk's split at each resume
