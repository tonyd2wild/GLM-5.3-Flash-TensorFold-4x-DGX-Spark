"""TF_GLM_PROMPT_OWN_FRONT=1: a row-split prompt chunk's DSA layers run their replicated front on each rank's own rows.

With the row split (``hcsplit``, TF_GLM_HC_SPLIT=1) each rank ends a layer holding the next layer's normed input for
its own rows [r H, r H + H) only, and swaps those bf16 rows (hidden 4,096: 8 KiB a row) so that every rank can run the
next block on every row. A KDA block and the MoE need every row (their projections are split by heads and experts),
but most of a DSA block's front is replicated, not head-split: q_a / kv_a (one projection, ``a.proj``) and their
RMSNorms, the indexer's key and weight projection (``kw``), its gate, its query projection (``qb``) and the top-k
selection over every visible pooled key, which grows with the context. Every rank ran all of it on every row.

Here, before a DSA layer, the glue keeps the normed rows on their owner (``HcSplit.glue(share=False)``); each rank runs
that front on its own rows and the ranks share its outputs instead:

1. own rows: q_a / kv_a and their norms (b.qr, b.lat) and q_b for this rank's heads (``forward.dsa_front``, already
   run during the glue with TF_GLM_PREFILL_OVERLAP=2), with the indexer the index keys and weights (b.ikr) and gates
   (b.igr), and when the chunk has sparse rows the index queries (b.qi);
2. shared: b.ikr and b.igr's rows (with the indexer), then b.qr and b.lat's rows (4 KiB a row);
3. every rank: every row's index key, gate and pooled key (``sparse.index_update``, as before: the same kernel on the
   same rows), then this rank's rows' selection over those keys (``sparse.select_pools``), whose pools (512 int32 a
   row) are shared in turn;
4. every rank, every row, as before: q_b for its heads on the other ranks' rows, the latent write, absorb, dense
   attention, then the tokens of every row from the pools (``sparse.pool_tokens``), sparse attention, expand and
   o_proj.

Every kernel is the unchanged path's on the same row (the projections, norms and writers keep rows apart; a row's
pools depend only on its own query and weights and the pooled keys at or before its position, which every rank has
written before any rank selects, and are the same in any block of rows: ``sparse.select_pools``), so the caches, the
sparse rows' tokens and every output are the unchanged path's bit for bit, and identical on every rank: what each rank
writes it computed from the same shared bytes. Rows that keep dense attention (count 0) may get other, unread, pools.

Order: with the overlap (TF_GLM_PREFILL_OVERLAP=1 or 2) the shares run on the split's side stream, in one order on
every rank (index keys, the rows, then the pools), so the index update and selection run while the rows are on the
wire, and q_b, the latent write, absorb and dense attention while the pools are. Without it they run in place.

Not for several streams' prompts in one chunk (``multi_prefill``'s mixer), chunks too short to split, decode and
verify windows, nor without the latent cache (TF_GLM_LATENT=0): those keep the unchanged path.

By Mia's AI Lab (Apache-2.0), ported from the GLM-5.3 recipe's patch 0147 to Flash's DSA (pooled keys, no rotary
part). Design credit: drowzeys (github.com/drowzeys/TensorFold, branch glm53-tp4-2026-10-05, ``compute_prompt_sp`` with
TF_GLM53_SP_SELECT) authored the sequence-parallel prompt front this follows (the replicated projections and the top-k
on a rank's own rows, their outputs gathered); no code of theirs."""

from __future__ import annotations

import torch

from . import glue, latent, prof, sparse


def applies(w, st, b, mixer, host_pos) -> bool:
    """Whether a row-split prompt chunk's DSA layers take the own-row front: TF_GLM_PROMPT_OWN_FRONT, one stream's
    prompt on ``st`` (no ``mixer``), its position known on the host, the latent cache."""

    sp = b.split
    return (sp is not None and sp.settings.own_front and mixer is None and st is not None and host_pos is not None
            and b.prefill and latent.ENABLED)


def takes(layer) -> bool:
    """A layer whose attention block runs here: DSA on the latent cache."""
    return layer.kind != "kda" and layer.dsa.absorb is not None


def _lists(b):
    """The chunk's pools [alloc rows, TOPK_POOLS], tokens [alloc rows, TOKENS] and counts [alloc rows], int32, kept
    with the prompt buffers (each rank writes its own rows' pools, the shares the others'; every rank the tokens)."""

    t = getattr(b, "own_lists", None)
    if t is None:
        dev, rows = b.ids.device, b.alloc_rows
        t = (torch.empty((rows, sparse.TOPK_POOLS), dtype=torch.int32, device=dev),
             torch.empty((rows, sparse.TOKENS), dtype=torch.int32, device=dev),
             torch.empty((rows,), dtype=torch.int32, device=dev))
        b.own_lists = t
    return t


def _others(R: int, lo: int, hi: int) -> list[tuple[int, int]]:
    """The row ranges of [0, R) outside this rank's [lo, hi)."""

    return [(a, z) for a, z in ((0, min(lo, R)), (max(hi, 0), R)) if z > a]


class Front:
    """A layer's own-row front of one chunk (``front``), what ``rest`` needs: the rows, the flags, the shares' events
    (None: they ran on the current stream)."""

    __slots__ = ("lo", "hi", "sparse_rows", "all_sparse", "ev_rows", "ev_pools")

    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def front(layer, w, st, b, R: int, host_pos: int, done: tuple[int, int] | None = None) -> Front:
    """Steps 1 to 3 on the current stream: this rank's rows' replicated front, its shares, every row's index keys,
    this rank's rows' selection and its share. ``done``: rows ``forward.dsa_front`` already ran on (the glue's
    fronts, TF_GLM_PREFILL_OVERLAP=2)."""

    from .forward import dsa_front, mm

    c = w.cfg
    a = layer.dsa
    sp = b.split
    ix = a.index
    di = st.dsa_index[layer.index]
    index = st.index[di] if st.index is not None else None
    lo, hi = sp.own()
    n = max(0, hi - lo)
    all_sparse = host_pos >= c.dense_limit
    sparse_rows = index is not None and host_pos + R - 1 >= c.dense_limit

    # 1. the replicated front on this rank's rows
    with prof.timed("dsa: own front"):
        if n:
            if done != (lo, hi):
                dsa_front(layer, w, b, lo, hi)
            if index is not None:
                mm(b, b.normed[lo:hi], ix.kw, b.xs[lo:hi], b.ikr[lo:hi])
                glue.router(b.normed[lo:hi], ix.gate, b.igr[lo:hi])
            if sparse_rows:
                mm(b, b.qr[lo:hi], ix.qb, b.xs_qr[lo:hi], b.qi[lo:hi])
    # 2. shared: the index keys and gates first (the index update waits for them only), then the norms' rows
    ev_keys = sp.share([b.ikr, b.igr], slot=0) if index is not None else None
    ev_rows = sp.share([b.qr, b.lat], slot=1)
    # 3. every row's index keys, then this rank's rows' pools, shared
    ev_pools = None
    if index is not None:
        ik, ig, pk = index
        sp.wait(ev_keys)
        with prof.timed("dsa: indexer update"):
            sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, st.pos_dev)
        if sparse_rows:
            pools, _, _ = _lists(b)
            with prof.timed("dsa: select tokens"):
                if n:
                    at = st.pos_dev + lo if lo else st.pos_dev
                    sparse.select_pools(b.qi[lo:hi], b.ikr[lo:hi, c.index_dim:], pk, host_pos + lo, n,
                                        pk.shape[0] - 2, at, pools[lo:hi])
            ev_pools = sp.share([pools], slot=2)
    return Front(lo=lo, hi=hi, sparse_rows=sparse_rows, all_sparse=all_sparse, ev_rows=ev_rows, ev_pools=ev_pools)


def rest(layer, w, st, b, R: int, nch: int | None, host_pos: int, f: Front):
    """Step 4 on the current stream after ``front``: every row's attention for this rank's heads, kv_b's value rows
    and o_proj; this rank's partials go out through the split (``forward.partials``: None returned)."""

    from . import l2pf, qmm
    from .forward import dense_attention, dense_rows, mm, out_proj, sparse_prompt

    c = w.cfg
    a = layer.dsa
    sp = b.split
    lc = st.kc[st.dsa_index[layer.index]]
    pos_dev = st.pos_dev
    HL = a.heads
    # 4. every row: q_b on the other ranks' rows, the latent write, the queries
    sp.wait(f.ev_rows)
    with prof.timed("dsa: projections"):
        for x, z in _others(R, f.lo, f.hi):
            mm(b, b.qr[x:z], a.q_b, b.xs_qr[x:z], b.q[x:z].view(z - x, HL * c.qk_dim))
    s = b.lat_s
    with prof.timed("dsa: latent write"):
        latent.latent_write(b.lat[:R], lc, pos_dev)
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[:R], a.absorb, s.qa[:R], prompt=latent.PROMPT_MMA)
    l2pf.site(layer.index, "o")
    ol = s.ol[:R]
    scale = c.qk_dim ** -0.5
    if not f.all_sparse:
        dense = dense_rows(c, R, host_pos) if f.sparse_rows else R
        with prof.timed("dsa: dense attention"):
            dense_attention(qa[:dense], lc, pos_dev, s, scale=scale, nch=min(nch or s.nch, s.nch), out=ol[:dense],
                            hb=latent.head_block(R), starts=b.att_starts)
    if f.sparse_rows:
        pools, tokens, counts = _lists(b)
        sp.wait(f.ev_pools)
        with prof.timed("dsa: select tokens"):
            sparse.pool_tokens(pools[:R], pos_dev, tokens[:R], counts[:R])
        with prof.timed("dsa: sparse attention"):
            sparse_prompt(qa, lc, tokens[:R], counts[:R], ol, scale)
    with prof.timed("dsa: expand"):
        o = latent.expand_v(ol, a.absorb, b.vn[:R], prompt=latent.PROMPT_MMA).view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm.group_sums(o, b.xs_ao[:R]), R, site=(layer.index, "a"))


def dsa_block(layer, w, st, b, R: int, nch: int | None, host_pos: int, done: tuple[int, int] | None = None):
    """A DSA block of a row-split prompt chunk with its replicated front on this rank's rows (the module's steps, all
    on the current stream); this rank's partials go out through the split (None returned)."""

    return rest(layer, w, st, b, R, nch, host_pos, front(layer, w, st, b, R, host_pos, done))
