"""TF_GLM_HC_SPLIT=1: a prompt chunk's hyper-connection glue split by rows between the ranks (decode untouched).

Unsplit, each exchange site (after the attention block and after the MLP/MoE block) all-gathers both ranks' fp32
partials [R, D], and both ranks run hc_post and the next hc_pre over all R rows. Split, rank r owns rows
[r H, r H + H) (H = R / 2; an odd chunk gets one zero pad row, which nothing reads): each rank sends the peer its
partial of the peer's rows (fp32), adds the two partials of its own rows rank 0's first (``glue.hc_post_pair``, the
same kernel as hc_post), runs hc_post and the next hc_pre on its own rows, and swaps its normed rows (bf16) for the
peer's, so both ranks hold the next block's full input. The residual streams (b.x) stay current for own rows only;
DFlash2 taps and the final stream mean (b.hidden) are computed on own rows and swapped like the normed rows. Every
glue kernel is row-independent and the partial sum order is unchanged, so every bit is the unsplit path's.

The group sums b.xs of the normed rows are not swapped: no prompt-chunk matmul reads them (4-bit projections of a
prompt chunk run ``prefill_matmul``, BF16 and FP8 ones ignore them). The tests compare split and unsplit states bit
for bit, so a later reader would show up there.

TF_GLM_PREFILL_OVERLAP=1 (with the split) runs the swaps on a second CUDA stream in row pieces: a site computes the
peer's rows of its partial first, piece by piece, each piece's swap starting as it is done, then its own rows; then
the glue piece by piece, each piece waiting only for its own partial rows and its normed rows swapped while the next
piece's glue runs. Events order the two streams; the arithmetic is the same kernels on the same rows.

TF_GLM_PREFILL_OVERLAP=2 also runs the next block's row-independent front (``glue``'s ``front``: KDA's input
projections, DSA's projections, norms and query expansion, the dense MLP's gate/up, the shared expert of the MoE) on
this rank's rows of each piece right after that piece's glue, while its normed rows (and the later pieces' partials)
are still on the wire; the block then runs its front on the peer's rows once they have arrived, and the rest as
before. Without it the main stream waited for the last piece's swap with nothing to do. The fronts are row-independent
kernels (the prompt matmul gives every row range the same bits), so every bit is still the unsplit path's.

TF_GLM_PROMPT_OWN_FRONT=1 (``own_front``): the glue before a DSA layer keeps the normed rows on their owner, whose
attention block runs its replicated front (q_a / kv_a and their norms, the indexer's projections and its top-k) on
them and shares the outputs (``share``: whole own shares, send/receive or the all-gather, on the side stream with the
overlap) instead.

TF_GLM_PROMPT_MICROBATCH=1 (``microbatch``): a chunk runs as two halves, each row-split by its own HcSplit over row
views of the prompt buffers, both on one side stream (``HcSplit(stream=...)``); a half's glue runs there too
(``glue_side``), under the other half's compute, and the main stream ``join``s it before that half's next block.

TF_GLM_PROMPT_PARTIALS=bf16 (``partials``; default fp32): each rank's partial of a site goes out as bf16 instead of fp32
(half the bytes of a layer's two partial exchanges, most of what a row-split layer sends), and every rank's partial of a
row is rounded to bf16 before the sum, which stays in fp32 and rank 0 first: the owner rounds its own partial as the
others round theirs (round-to-nearest-even in torch's casts and the kernels', the same), so the ranks hold the same
bits. Unsplit prompt chunks (a prompt's tail shorter than TF_GLM_HC_SPLIT_MIN_ROWS) and the all-gather exchange round
the same way in hc_post (``glue.hc_post``'s ``round_parts``), so a row's bits do not depend on the chunking, the
micro-batches or the exchange; decode windows, verify and the MTP head keep fp32 partials. The prompt's bits change
once against fp32 partials (each partial rounded to bf16 before the sum; the sum is rounded to bf16 as before), which
has not been quality-checked on Flash: opt-in.

TF_GLM_HC_EXCHANGE: ``p2p`` (the default) swaps rows with ncclSend/ncclRecv; ``gather`` uses the NCCL all-gather
instead (whole halves, one piece: the partial halves out of place into the unused all-gather buffer, the bf16 rows in
place), the same bytes on the wire, for when NCCL's point-to-point path is slower or misbehaves on a fabric."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch

from . import glue

EXCHANGES = ("p2p", "gather")
PARTIALS = ("fp32", "bf16")    # TF_GLM_PROMPT_PARTIALS: what a row-split prompt chunk's partials travel as
SHARES = 4               # own-row shares a layer keeps in flight at once (``HcSplit.share``: slots 0 .. SHARES - 1)


def _flag(env, name: str) -> bool:
    value = str(env.get(name, "") or "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name}: 0 or 1, not {value!r}")
    return value == "1"


def _int(env, name: str, default: int, lo: int, hi: int) -> int:
    value = str(env.get(name, "") or default).strip()
    if not value.isdecimal() or not lo <= int(value) <= hi:
        raise ValueError(f"{name}: a whole number from {lo} to {hi}, not {value!r}")
    return int(value)


def pad_rows(rows: int, world: int) -> int:
    """Rows a prompt buffer of ``rows`` rows holds past them for the split's pad rows: a chunk of ``rows`` rows rounded
    up to whole shares a rank (``HcSplit.padded``), so that a full chunk, and every shorter one, runs split at any
    world size. Prompt chunks are a multiple of 64 rows (TF_GLM_PREFILL_ROWS), so this is 0 wherever the world size
    divides 64 (two and four ranks: 2,048 = 2 x 1,024 = 4 x 512) and ``-rows % 3`` at three (2,048 = 3 x 683 - 1:
    one row; 2,112: none). Without that row every full chunk at three ranks failed ``applies`` and ran unsplit (an
    all-gather of every rank's whole partial at each of the 90 sites, the glue on every row, no overlap)."""

    return -int(rows) % int(world)


def buffer_rows(rows: int, world: int, settings: "SplitSettings | None") -> int:
    """Rows the prompt buffers of ``rows``-row chunks hold (``forward.Buffers``' ``rows + pad``): ``rows`` plus the
    split's pad rows when TF_GLM_HC_SPLIT is on past one rank, else ``rows``. The engine's buffers and its memory
    estimate both take it, so they agree; at two and four ranks it is ``rows`` (the same buffers and estimate)."""

    if settings is None or not settings.split or int(world) < 2:
        return int(rows)
    return int(rows) + pad_rows(rows, world)


@dataclass(frozen=True)
class SplitSettings:
    """What TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP ask for; both ranks must agree (the startup comparison)."""

    split: bool = False
    overlap: bool = False
    pieces: int = 2               # TF_GLM_OVERLAP_PIECES: row pieces a site's swaps run in (overlap, p2p only)
    min_rows: int = 64            # TF_GLM_HC_SPLIT_MIN_ROWS: shorter chunks (a prompt's tail) run unsplit
    exchange: str = "p2p"         # TF_GLM_HC_EXCHANGE: p2p (ncclSend/ncclRecv) or gather (the all-gather)
    fronts: bool = False          # TF_GLM_PREFILL_OVERLAP=2: the next block's front on own rows during the swaps
    own_front: bool = False       # TF_GLM_PROMPT_OWN_FRONT=1: DSA's replicated front on own rows (``own_front``)
    microbatch: bool = False      # TF_GLM_PROMPT_MICROBATCH=1: a chunk in two halves (``microbatch``)
    micro_min: int = 2048         # TF_GLM_PROMPT_MICROBATCH_MIN_ROWS: halves of fewer rows run the chunk whole
    partials: str = "fp32"        # TF_GLM_PROMPT_PARTIALS: fp32, or bf16 (rounded before the sum, half the bytes)

    def __post_init__(self) -> None:
        if self.fronts and not self.overlap:
            raise ValueError("the blocks' fronts run during the overlapped swaps: they need the overlap")
        if self.own_front and not self.split:
            raise ValueError("TF_GLM_PROMPT_OWN_FRONT=1 runs on the row split's own rows: it needs TF_GLM_HC_SPLIT=1")
        if self.microbatch and not self.split:
            raise ValueError("TF_GLM_PROMPT_MICROBATCH=1 row-splits each half: it needs TF_GLM_HC_SPLIT=1")
        if self.partials not in PARTIALS:
            raise ValueError(f"TF_GLM_PROMPT_PARTIALS: {' or '.join(PARTIALS)}, not {self.partials!r}")
        if self.partials != "fp32" and not self.split:
            raise ValueError("TF_GLM_PROMPT_PARTIALS=bf16 sends the row split's partials as bf16: it needs "
                             "TF_GLM_HC_SPLIT=1")

    @classmethod
    def from_env(cls, env=None) -> "SplitSettings":
        env = os.environ if env is None else env
        split = _flag(env, "TF_GLM_HC_SPLIT")
        level = str(env.get("TF_GLM_PREFILL_OVERLAP", "") or "0").strip()
        if level not in ("0", "1", "2"):
            raise ValueError(f"TF_GLM_PREFILL_OVERLAP: 0, 1 or 2, not {level!r}")
        overlap = level != "0"
        if overlap and not split:
            raise ValueError(f"TF_GLM_PREFILL_OVERLAP={level} overlaps the row-split exchanges: it needs "
                             "TF_GLM_HC_SPLIT=1")
        exchange = str(env.get("TF_GLM_HC_EXCHANGE", "") or "p2p").strip()
        if exchange not in EXCHANGES:
            raise ValueError(f"TF_GLM_HC_EXCHANGE: {' or '.join(EXCHANGES)}, not {exchange!r}")
        partials = str(env.get("TF_GLM_PROMPT_PARTIALS", "") or "fp32").strip()
        return cls(split, overlap, _int(env, "TF_GLM_OVERLAP_PIECES", 2, 1, 8),
                   _int(env, "TF_GLM_HC_SPLIT_MIN_ROWS", 64, 2, 16384), exchange, fronts=level == "2",
                   own_front=_flag(env, "TF_GLM_PROMPT_OWN_FRONT"),
                   microbatch=_flag(env, "TF_GLM_PROMPT_MICROBATCH"),
                   micro_min=_int(env, "TF_GLM_PROMPT_MICROBATCH_MIN_ROWS", 2048, 64, 16384), partials=partials)

    def piece_count(self) -> int:
        return self.pieces if self.overlap and self.exchange == "p2p" else 1

    def code(self) -> list[int]:
        if not self.split:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        return [1, int(self.overlap) + int(self.fronts), self.piece_count(), self.min_rows,
                EXCHANGES.index(self.exchange), int(self.own_front), self.micro_min if self.microbatch else 0,
                PARTIALS.index(self.partials)]

    def describe(self) -> str:
        what = "rows split between the ranks"
        what += ", exchanges by " + ("send/receive" if self.exchange == "p2p" else "all-gather")
        if self.overlap:
            n = self.piece_count()
            what += f" on a second stream{f' in {n} pieces' if n > 1 else ''}"
            if self.fronts:
                what += ", the next block's front on own rows meanwhile"
        if self.own_front:
            what += "; DSA's replicated front (q_a / kv_a, the indexer, its top-k) on own rows, its outputs shared"
        if self.microbatch:
            what += f"; chunks of {2 * self.micro_min}+ rows in two halves, one's exchanges under the other's compute"
        if self.partials == "bf16":
            what += "; partials rounded to bf16 before the sum and sent as bf16 (TF_GLM_PROMPT_PARTIALS)"
        return f"prompt chunks' hyper-connections: {what} (chunks of {self.min_rows}+ rows)"


class HcSplit:
    """One prompt buffer's row split: rows, pieces, the second stream and its events.

    Past two ranks (``world`` N): rank r owns rows [r H, r H + H), H = ceil(R / N) (pad rows zero, never read).
    p2p: each rank sends every peer its partial of that peer's rows and receives each peer's partial of its own rows
    into a [N, n, D] staging block a piece (its own partial copied into its slot), summed rank 0 first by hc_post
    (``exchange_all``: one NCCL group, every pair a send and a receive, so it needs every pair connected); its glued
    rows go to every peer the same way. gather (for fabrics without every pair, e.g. a ring): the unsplit all-gather
    of every rank's whole partial, glue on own rows (``glue.hc_post_ranks``), then the in-place all-gather of the
    rows. Either way the partial sums are hc_post's, rank 0 first: the unsplit bits."""

    def __init__(self, w, b, settings: SplitSettings, stream=None) -> None:
        world = int(getattr(w, "world", 2))
        two = world == 2
        p2p = settings.exchange == "p2p"
        if world < 2 or w.comm is None or (p2p and not hasattr(w.comm, "exchange" if two else "exchange_all")):
            raise ValueError("TF_GLM_HC_SPLIT needs two ranks or more and a communicator with exchange "
                             "(send/receive), or TF_GLM_HC_EXCHANGE=gather")
        self.w, self.b, self.settings = w, b, settings
        self.world = world
        self.rank = w.rank
        self.peer = 1 - w.rank if two else None
        self.peers = [r for r in range(world) if r != w.rank]
        self.gather = not p2p
        # TF_GLM_PROMPT_PARTIALS=bf16: every partial rounded to bf16 before the sum (``round``); with p2p also sent as
        # bf16 (``half``: staged in the all-gather buffer's unused part, ``begin``). The all-gather exchange sends fp32
        # and rounds in hc_post: the same bits
        self.round = settings.partials == "bf16"
        self.half = self.round and p2p
        self.send16: torch.Tensor | None = None
        # the plain NCCL all-gather (RoceComm sends small gathers over RoCE, whose kernel is not written for in place)
        self.all_gather = getattr(w.comm, "nccl", w.comm).all_gather
        self.active = False
        self.R = self.Rp = self.H = 0
        self.pieces: list[tuple[int, int]] = []
        self.got: list[torch.Tensor] = []
        self.stream = None
        self.pending = False             # a glue_side queued on the side stream that the main stream has not joined
        self.micro: list = []            # TF_GLM_PROMPT_MICROBATCH: the halves' splits of the chunk running now
        if settings.overlap:
            # TF_GLM_OVERLAP_PRIORITY: the side stream's priority, 0 (normal, the default) or -1 (high). Two Sparks, two
            # boots each: prefill the same, decode rounds after a high-priority prompt ~0.5-1% slower. ``stream``: one
            # to share (TF_GLM_PROMPT_MICROBATCH: both halves' splits on the buffers' one side stream)
            high = os.environ.get("TF_GLM_OVERLAP_PRIORITY", "0").strip() == "-1"
            self.stream = stream if stream is not None else torch.cuda.Stream(priority=-1 if high else 0)
            self.ev_own = torch.cuda.Event()
            n = settings.piece_count()
            self.ev_fill = [torch.cuda.Event() for _ in range(n)]
            self.ev_part = [torch.cuda.Event() for _ in range(n)]
            self.ev_glue = [torch.cuda.Event() for _ in range(n)]
            self.ev_done = torch.cuda.Event()
            # TF_GLM_PROMPT_OWN_FRONT: a layer's own-row shares (``share``): up to SHARES in flight on the side stream
            self.ev_share_in = [torch.cuda.Event() for _ in range(SHARES)]
            self.ev_share_out = [torch.cuda.Event() for _ in range(SHARES)]

    def padded(self, R: int) -> int:
        """R rounded up to a whole share a rank (two ranks: an odd chunk gets one pad row)."""
        return -(-R // self.world) * self.world

    def applies(self, R: int) -> bool:
        """Whether a chunk of R rows runs split: long enough, and its pad rows fit the buffers (``pad_rows``)."""
        return R >= self.settings.min_rows and self.padded(R) <= getattr(self.b, "alloc_rows", self.b.rows)

    def warm(self) -> None:
        """Open the NCCL send/receive connections to the peers at startup (every rank calls this together)."""

        if self.gather:
            return
        a = torch.zeros((1,), dtype=torch.float32, device="cuda")
        z = torch.empty_like(a)
        if self.world == 2:
            self.w.comm.exchange([a], [z], self.peer)
        else:
            outs = {p: [torch.empty_like(a)] for p in self.peers}
            self.w.comm.exchange_all({p: [a] for p in self.peers}, outs)
        torch.cuda.synchronize()

    # -- per chunk ------------------------------------------------------------------------------------------------
    def begin(self, R: int) -> None:
        b = self.b
        self.R, self.Rp = R, self.padded(R)
        self.H = H = self.Rp // self.world
        n = self.settings.piece_count()
        size = -(-H // n)
        if size > 64:                                 # whole 64-row tiles for the glue kernels where possible
            size = min(H, -(-size // 64) * 64)
        self.pieces = [(off, min(size, H - off)) for off in range(0, H, size)]
        # where the peers' partials of each piece of this rank's rows land, in the all-gather buffer (unused while
        # split). Two ranks: [n, D] fp32 a piece (gather: the peer's slot of the piece's [2, n, D]); more: a [N, n, D]
        # block a piece (p2p), or every rank's whole partial [N, Rp, D] (gather, one piece)
        d = b.part.shape[1]
        N = self.world
        if self.half:
            # bf16: the peers' partials of this rank's rows first ([n, D] a piece at two ranks, [N, n, D] past two:
            # Rp x D values in all), then this rank's bf16 partial of the peers' rows (``send16``, by row: Rp x D),
            # in the all-gather buffer's room for world x rows x D fp32
            g16 = b.gath.view(torch.bfloat16)
            if 2 * self.Rp * d > g16.numel():
                raise ValueError("TF_GLM_PROMPT_PARTIALS=bf16: the all-gather buffer cannot hold the chunk's staging")
            if N == 2:
                self.got = [g16[off * d:(off + n) * d].view(n, d) for off, n in self.pieces]
            else:
                self.got = [g16[N * off * d:N * (off + n) * d].view(N, n, d) for off, n in self.pieces]
            self.send16 = g16[self.Rp * d:2 * self.Rp * d].view(self.Rp, d)
        elif N == 2 and self.gather:
            self.got = [b.gath[(2 * off + self.peer * n) * d:(2 * off + self.peer * n + n) * d].view(n, d)
                        for off, n in self.pieces]
        elif N == 2:
            self.got = [b.gath[off * d:(off + n) * d].view(n, d) for off, n in self.pieces]
        elif self.gather:
            self.got = [b.gath[:N * self.Rp * d].view(N, self.Rp, d)]
        else:
            self.got = [b.gath[N * off * d:N * (off + n) * d].view(N, n, d) for off, n in self.pieces]
        if self.Rp > R:                                # the pad rows: zero streams and partials, never read
            b.x[R:self.Rp].zero_()
            b.part[R:self.Rp].zero_()
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())
        self.active = True

    def finish(self) -> None:
        if self.stream is not None and self.active:
            torch.cuda.current_stream().wait_stream(self.stream)
        self.active = False
        self.pending = False
        for half in self.micro:          # a halved chunk's splits (a layer-sliced yield): their side work joined, they
            half.drain()                 # stay active for the chunk's next layers

    def drain(self) -> None:
        """The current stream waits for everything queued on the side stream; the split stays active."""
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        self.pending = False

    def join(self) -> None:
        """The current stream waits for this split's ``glue_side`` (nothing queued: nothing to wait for)."""
        if self.pending:
            torch.cuda.current_stream().wait_event(self.ev_done)
            self.pending = False

    def mine(self) -> int:
        return self.rank * self.H

    def theirs(self) -> int:
        return self.peer * self.H

    def _fill(self, fill: Callable[[int, int], None], lo: int, hi: int) -> None:
        hi = min(hi, self.R)
        if hi > lo:
            fill(lo, hi)

    def _swap_partial(self, j: int) -> None:
        """Piece j: this rank's partial of the peers' rows out, the peers' partials of this rank's rows in."""

        off, n = self.pieces[j]
        b = self.b
        if self.world > 2:
            if self.gather:                             # every rank's whole partial (one piece)
                self.all_gather(b.part[:self.Rp].reshape(-1), self.got[0].reshape(-1))
                return
            if self.half:                               # each peer's rows rounded to bf16 (on this stream)
                sends = {}
                for p in self.peers:
                    lo = p * self.H + off
                    sends[p] = [self.send16[lo:lo + n]]
                    sends[p][0].copy_(b.part[lo:lo + n])
            else:
                sends = {p: [b.part[p * self.H + off:p * self.H + off + n]] for p in self.peers}
            self.w.comm.exchange_all(sends, {p: [self.got[j][p]] for p in self.peers})
            return
        lo = self.theirs() + off
        send = b.part[lo:lo + n]
        if self.half:
            send = self.send16[lo:lo + n]
            send.copy_(b.part[lo:lo + n])
        if self.gather:
            d = send.shape[1]
            self.all_gather(send.reshape(-1), b.gath[2 * off * d:2 * (off + n) * d])
        else:
            self.w.comm.exchange([send], [self.got[j]], self.peer)

    def _swap_rows(self, j: int, outs: list[torch.Tensor]) -> None:
        """Piece j of bf16 row buffers: this rank's rows out, the peers' rows in (gather: in place, whole shares)."""

        off, n = self.pieces[j]
        mine = self.mine() + off
        if self.gather:                                 # one piece: rank order is row order
            for t in outs:
                self.all_gather(t[mine:mine + n].reshape(-1), t[:self.Rp].reshape(-1))
            return
        if self.world > 2:
            self.w.comm.exchange_all({p: [t[mine:mine + n] for t in outs] for p in self.peers},
                                     {p: [t[p * self.H + off:p * self.H + off + n] for t in outs] for p in self.peers})
            return
        theirs = self.theirs() + off
        self.w.comm.exchange([t[mine:mine + n] for t in outs], [t[theirs:theirs + n] for t in outs], self.peer)

    def _swap_share(self, outs: list[torch.Tensor]) -> None:
        """Every rank's whole share of rows [r H, r H + H) of each row buffer in ``outs`` to every rank (this rank's
        out, the peers' in; pad rows travel too, never read): ``_swap_rows`` for the share in one piece."""

        H, mine = self.H, self.mine()
        if self.gather:                                 # rank order is row order
            for t in outs:
                self.all_gather(t[mine:mine + H].reshape(-1), t[:self.Rp].reshape(-1))
            return
        if self.world > 2:
            self.w.comm.exchange_all({p: [t[mine:mine + H] for t in outs] for p in self.peers},
                                     {p: [t[p * H:p * H + H] for t in outs] for p in self.peers})
            return
        theirs = self.theirs()
        self.w.comm.exchange([t[mine:mine + H] for t in outs], [t[theirs:theirs + H] for t in outs], self.peer)

    def share(self, outs: list[torch.Tensor], slot: int = 0):
        """TF_GLM_PROMPT_OWN_FRONT: this rank's rows of ``outs`` (contiguous row buffers it wrote on the current
        stream) to every peer and theirs in. With the overlap: on the side stream after the current stream's work so
        far, returning the event the reader waits on (``wait``) before it reads the peers' rows; it must not write
        ``outs`` before that either. Without: on the current stream, None. Every rank issues its shares (and the
        glue's swaps) in one order, so the collectives match."""

        if self.stream is None:
            self._swap_share(outs)
            return None
        main = torch.cuda.current_stream()
        ev_in, ev_out = self.ev_share_in[slot], self.ev_share_out[slot]
        ev_in.record(main)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ev_in)
            self._swap_share(outs)
            ev_out.record(self.stream)
        return ev_out

    @staticmethod
    def wait(ev) -> None:
        """The current stream waits for a ``share`` (None: it ran on the current stream)."""
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)

    def partial(self, fill: Callable[[int, int], None]) -> None:
        """This rank's fp32 partial rows (``fill(lo, hi)`` writes b.part[lo:hi]), the peers' rows sent to them and
        their partials of this rank's rows received (``got``)."""

        mine = self.mine()
        if self.stream is None or (self.world > 2 and self.gather):
            fill(0, self.R)
            if self.stream is None:
                for j in range(len(self.pieces)):
                    self._swap_partial(j)
                return
            main = torch.cuda.current_stream()          # past two ranks, gather: the whole partial, then its gather
            self.ev_fill[0].record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.ev_fill[0])
                self._swap_partial(0)
                self.ev_part[0].record(self.stream)
            return
        main = torch.cuda.current_stream()
        others = [self.theirs()] if self.world == 2 else [p * self.H for p in self.peers]
        for j, (off, n) in enumerate(self.pieces):
            for start in others:
                self._fill(fill, start + off, start + off + n)
            self.ev_fill[j].record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.ev_fill[j])
                self._swap_partial(j)
                self.ev_part[j].record(self.stream)
        self._fill(fill, mine, mine + self.H)

    def own(self) -> tuple[int, int]:
        """This rank's rows of the chunk, [lo, hi) (the pad rows left out)."""

        return self.mine(), min(self.mine() + self.H, self.R)

    def _post(self, j: int, lo: int, hi: int) -> None:
        """hc_post of this rank's rows lo .. hi (piece j): every rank's partial of them summed rank 0 first."""

        b = self.b
        x = b.x[lo:hi]
        own = b.part[lo:hi]
        if self.world == 2:
            got = self.got[j]
            if self.half:
                own = own.to(torch.bfloat16)
            g0, g1 = (own, got) if self.rank == 0 else (got, own)
            glue.hc_post_pair(x, x, g0, g1, b.post[lo:hi], b.comb[lo:hi], round_parts=self.round)
        elif self.gather:
            whole = self.got[0]
            glue.hc_post_ranks(x, x, whole[0, lo:hi], self.Rp * whole.shape[2], self.world, b.post[lo:hi],
                               b.comb[lo:hi], round_parts=self.round)
        else:
            block = self.got[j]
            block[self.rank].copy_(own)                 # bf16: rounded as the peers rounded theirs
            glue.hc_post(x, x, block, b.post[lo:hi], b.comb[lo:hi], round_parts=self.round)

    def glue_side(self, hc=None, norm=None, taps: tuple[int, ...] = (), final: bool = False,
                  share: bool = True) -> None:
        """``glue`` with its kernels on the side stream after the current stream's fills so far (TF_GLM_PROMPT_
        MICROBATCH: the half's glue under the other half's compute; ``join`` before reading what it wrote). The same
        kernels on the same rows in the same order, after the partials' exchange (earlier on the side stream). Without
        a side stream: ``glue`` in place."""

        if self.stream is None:
            self.glue(hc, norm, taps, final, share=share)
            return
        self.ev_own.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ev_own)
            self._glue_rows(hc, norm, taps, final, share)
            self.ev_done.record(self.stream)
        self.pending = True

    def _glue_rows(self, hc, norm, taps: tuple[int, ...], final: bool, share: bool) -> None:
        """``glue``'s work on the current stream, piece by piece, its swaps in place (``glue_side``)."""

        b, c = self.b, self.w.cfg
        mine = self.mine()
        outs = [b.taps[slot] for slot in taps] + ([b.hidden] if final else []) + \
            ([b.normed] if hc is not None and share else [])
        for j, (off, n) in enumerate(self.pieces):
            lo, hi = mine + off, mine + off + n
            self._post(j, lo, hi)
            x = b.x[lo:hi]
            for slot in taps:
                glue.stream_mean(x, b.taps[slot][lo:hi])
            if final:
                glue.stream_mean(x, b.hidden[lo:hi])
            if hc is not None:
                glue.hc_pre(x, hc.fn, hc.base, hc.scale, norm, b.normed[lo:hi], b.xs[lo:hi], b.post[lo:hi],
                            b.comb[lo:hi], b.hcpart[lo:hi], c.eps, c.hc_eps, c.hc_iters, prompt=True)
            if outs:
                self._swap_rows(j, outs)

    def glue(self, hc=None, norm=None, taps: tuple[int, ...] = (), final: bool = False,
             front: Callable[[int, int], None] | None = None, share: bool = True) -> None:
        """hc_post of this rank's rows, then (in order) the layer's DFlash2 taps, the final stream mean (``final``)
        and the next hc_pre (``hc``, ``norm``); their rows swapped with the peers'. ``front(lo, hi)``
        (TF_GLM_PREFILL_OVERLAP=2): the next block's row-independent start on this rank's rows of each piece, on the
        main stream right after the piece's glue, while its swap (and the later pieces' partials) are on the wire;
        the caller runs it on the peers' rows (``own`` is what was done) after this returns. ``share`` False
        (TF_GLM_PROMPT_OWN_FRONT, before a DSA layer): the normed rows stay this rank's own (the DSA block's own-row
        front shares what the other ranks need); the taps and the final mean still travel."""

        b, c = self.b, self.w.cfg
        mine = self.mine()
        main = torch.cuda.current_stream()
        outs = [b.taps[slot] for slot in taps] + ([b.hidden] if final else []) + \
            ([b.normed] if hc is not None and share else [])
        for j, (off, n) in enumerate(self.pieces):
            lo, hi = mine + off, mine + off + n
            if self.stream is not None:
                main.wait_event(self.ev_part[0 if self.world > 2 and self.gather else j])
            self._post(j, lo, hi)
            x = b.x[lo:hi]
            for slot in taps:
                glue.stream_mean(x, b.taps[slot][lo:hi])
            if final:
                glue.stream_mean(x, b.hidden[lo:hi])
            if hc is not None:
                glue.hc_pre(x, hc.fn, hc.base, hc.scale, norm, b.normed[lo:hi], b.xs[lo:hi], b.post[lo:hi],
                            b.comb[lo:hi], b.hcpart[lo:hi], c.eps, c.hc_eps, c.hc_iters, prompt=True)
            if self.stream is None:
                if outs:
                    self._swap_rows(j, outs)
                if front is not None:
                    self._fill(front, lo, hi)
                continue
            if outs:
                self.ev_glue[j].record(main)
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(self.ev_glue[j])
                    self._swap_rows(j, outs)
            if front is not None:
                self._fill(front, lo, hi)
        if self.stream is not None:
            self.ev_done.record(self.stream)
            main.wait_event(self.ev_done)
