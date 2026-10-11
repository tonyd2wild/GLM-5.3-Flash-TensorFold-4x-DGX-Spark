"""GLM-5.3-Flash's concurrent requests (--parallel N) on two ranks: every stream commits exactly its own rows, so each
reply is the one its request gets served alone with ``"draft": false``.

Layout: the per-token caches are one pool (``pool.py``, ``forward.Caches``); a stream, or a kept prompt, owns an
extent of it aligned to 2,048 tokens, and the single-stream code (prompt chunks, verify windows, commits, snapshots)
runs on a ``forward.State`` viewing that extent and the stream's slot (``forward.Slots``: KDA states, conv windows,
window scratch; a DFlash2 context, ``dflash2_multi.DraftContext``; host side its copy-draft index, sampler and grammar).

Iterations (``GlmScheduler``, one worker thread on rank 0): admissions, then either one prompt chunk of the oldest
filling stream (FIFO, at most TF_GLM_FILL_ROWS rows while others decode, the engine's prompt chunk alone) or one
decode round, alternating by TF_GLM_FILL_SHARE; a prompt chunk never shares a forward with decode rows. TF_GLM_FILL_BUDGET_MS>0
yields between layer slices of a busy fill, retaining its prompt buffers while a decode round runs. With
TF_GLM_MULTI_PREFILL=1 a prompt chunk holds the next rows of several filling streams (``multi_prefill``: each
stream's rows get the bits of a chunk of their own; the shortest remaining prompts first, within the same row
budget). A round: each
decoding stream (admission order) proposes drafts under its own DFlash2 policy (copy drafts first), the combined
window is capped at TF_GLM_MULTI_WINDOW rows (``MAX_WINDOW`` by default; the longest tails trimmed), one verify
over every stream's rows (``forward_streams``: ``verify.BatchedVerify``, one forward over every stream's rows through
the segmented KDA and DSA kernels, or ``verify.SerialVerify``, the reference running each window as its own
forward; TF_GLM_MULTI_VERIFY), one packed all-gather for the samplers, each stream accepts and commits its own.
Between prompt slices, a round takes one token per stream without drafts to keep its latency and the newcomer's
inserted work small;
TF_GLM_FILL_SHARE applies to unsliced fills, slices alternate with these short rounds. Contiguous layer ranges
are balanced within the budget without adding rounds; GPU layer cost, finalization and dispatch overhead are
learned separately so extra slices cannot inflate every layer cost.

Rank 0 decides everything that depends on time, clients or placement; rank 1 applies it. Each iteration rank 0 sends
ONE message (``GlmEngine._share``): ops ADMIT, EVICT, MOVE, GROW, FILL, MFILL, FILLPART, MFILLPART, ROUND, FINISH, IDLE, which both ranks apply
in order through the same code (rank 0 as it decides them). Rank 1 derives drafts itself and checks the depths rank 0
sent. An idle rank 1 waits on the doorbell (``GlmEngine._await_bell``), not in an all-gather. After an error the two
ranks cannot be trusted to agree: ``broken`` latches and every later call fails until both restart.

Multi-stream mode drafts with DFlash2 only: MTP policies are remapped to DFlash2 ones (drafts never change a reply)."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.server.errors import CapacityError  # TFCAP-0075
from tensorfold.cuda.streams import Stream, next_fill

from . import multi_prefill, sliced_fill
from .multi_tune import MultiSettings, RoundProfile, allocate, reach_of, sample_packed, scaled_confidence
from .pool import ALIGN, Pool, align_up
from .verify import BatchedVerify, Segment, SerialVerify, Verified

ADMIT, EVICT, MOVE, GROW, FILL, ROUND, FINISH, IDLE, SLOT, MFILL, FILLPART, MFILLPART, LOAD, FLUSH, SPILL, LOADED, \
    TOUCH = range(1, 18)
OPS = {ADMIT: "ADMIT", EVICT: "EVICT", MOVE: "MOVE", GROW: "GROW", FILL: "FILL", ROUND: "ROUND", FINISH: "FINISH",
       IDLE: "IDLE", SLOT: "SLOT", MFILL: "MFILL", FILLPART: "FILLPART", MFILLPART: "MFILLPART", LOAD: "LOAD",
       FLUSH: "FLUSH", SPILL: "SPILL", LOADED: "LOADED", TOUCH: "TOUCH"}


def watchdog_seconds(value: str | None = None) -> float:
    """TF_GLM_MULTI_WATCHDOG_S: seconds a rank-0 iteration (or a rank-1 message it is owed) may take before every
    thread's Python stack is dumped to the log (faulthandler; 0: off; default 300)."""

    value = os.environ.get("TF_GLM_MULTI_WATCHDOG_S", "") if value is None else value
    try:
        s = 300.0 if value.strip() == "" else float(value)
    except ValueError:
        s = -1.0
    if s < 0:
        raise ValueError(f"TF_GLM_MULTI_WATCHDOG_S: seconds (0: off), not {value!r}")
    return s


def watchdog_exits(value: str | None = None) -> bool:
    """TF_GLM_MULTI_WATCHDOG_EXIT: 1 ends the process after the watchdog's first stack dump (a stalled server then
    exits with code 1, so a supervisor can restart it, instead of holding its requests forever); 0 (default): dumps
    every TF_GLM_MULTI_WATCHDOG_S and keeps waiting."""

    raw = (os.environ.get("TF_GLM_MULTI_WATCHDOG_EXIT", "") if value is None else value).strip() or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"TF_GLM_MULTI_WATCHDOG_EXIT: 0 or 1, not {raw!r}")
    return raw == "1"


def _watch(seconds: float, what: str) -> None:
    if seconds > 0:
        import faulthandler
        import sys

        stop = watchdog_exits()
        try:                                              # the process's stderr (a test may have swapped sys.stderr)
            faulthandler.dump_traceback_later(seconds, repeat=not stop, file=sys.__stderr__, exit=stop)
        except (ValueError, OSError, AttributeError):     # no file descriptor to write to: no watchdog
            pass


def _unwatch(seconds: float) -> None:
    if seconds > 0:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()


SEAL_MOD = 2_147_483_647                 # a message's checksum, modulo this prime (int32 on the wire)


def seal(msg: Sequence[int], seq: int) -> list[int]:
    """Rank 0: an iteration's message with its sequence number and a checksum appended (``unseal`` on rank 1)."""

    return [*msg, seq, digest(msg, seq)]


def digest(msg: Sequence[int], seq: int) -> int:
    a = np.asarray(msg, dtype=np.int64) % SEAL_MOD
    w = np.arange(1, len(a) + 1, dtype=np.int64) % SEAL_MOD
    return int((int((a * w % SEAL_MOD).sum()) + seq * 1_000_003) % SEAL_MOD)


def unseal(msg: Sequence[int], seq: int, rank: int = 1) -> list[int]:
    """A follower (rank ``rank``, 1 or more): the message rank 0 sealed as number ``seq``, or a RuntimeError naming
    what differs: the ranks would otherwise apply different ops and wait on each other's collectives forever (issue
    #29's hangs had no error)."""

    if len(msg) < 2:
        raise RuntimeError(f"rank {rank}: message {seq} from rank 0 is {len(msg)} values long, too short to be sealed; "
                           "the ranks are out of step: restart them all")
    body, got_seq, got = list(msg[:-2]), int(msg[-2]), int(msg[-1])
    if got_seq != seq or got != digest(body, got_seq):
        raise RuntimeError(f"rank {rank}: expected rank 0's message {seq}, received number {got_seq} "
                           f"({'checksum matches' if got == digest(body, got_seq) else 'checksum differs'}, "
                           f"{len(body)} values): the ranks are out of step: restart them all")
    return body


LONE_LEFT = 16                            # a lone stream moves home only with at least this many tokens still to go
DONE, CANCELLED, REQUEUED = 0, 1, 2      # FINISH reasons
MAX_WINDOW = 32                          # rows of every stream's windows together in one round (TF_GLM_MULTI_WINDOW)
SHARED_SLOTS = 2                         # shared-prefix points an ADMIT carries (engine.SHARED_MOST)


# -- settings ------------------------------------------------------------------------------------------------------
def fill_rows(prefill_rows: int, grid: int = 0, value: str | None = None) -> int:
    """TF_GLM_FILL_ROWS: rows of a prompt chunk while other streams decode (default 1,024 or the engine's prompt
    chunk if smaller): a multiple of 64 and of the prompt grid, at most the prompt chunk (TF_GLM_PREFILL_ROWS)."""

    value = os.environ.get("TF_GLM_FILL_ROWS", "") if value is None else value
    value = value.strip()
    rows = min(1024, prefill_rows) if value == "" else int(value) if value.isdecimal() else -1
    if rows < 64 or rows % 64 or rows > prefill_rows or (grid and rows % grid):
        raise ValueError(f"TF_GLM_FILL_ROWS: a multiple of 64{f' and of the {grid}-token prompt grid' if grid else ''}"
                         f" up to the {prefill_rows}-row prompt chunk, not {value!r}")
    return rows


def fill_unit(grid: int) -> int:
    """Rows a prompt chunk cut short of a stream's next stop comes in multiples of (a multi-prompt chunk's share of
    the row budget): 64 and the prompt grid (as TF_GLM_FILL_ROWS), so every cut stays on the grid."""

    import math

    return math.lcm(64, grid) if grid else 64


def fill_share(value: str | None = None) -> float:
    """TF_GLM_FILL_SHARE: the share of iterations prompt chunks take while other streams decode (default 0.5: a
    chunk, a round, a chunk, ...); 1 fills every prompt before decoding on."""

    value = os.environ.get("TF_GLM_FILL_SHARE", "") if value is None else value
    try:
        share = 0.5 if value.strip() == "" else float(value)
    except ValueError:
        share = -1.0
    if not 0 < share <= 1:
        raise ValueError(f"TF_GLM_FILL_SHARE: a share above 0 and at most 1, not {value!r}")
    return share


def keep_per_chat(value: str | None = None) -> int:
    """TF_GLM_KEEP_PER_CHAT: the kept states one conversation holds at its own turn boundaries (0, the default: no limit, what
    TensorFold does). Only that conversation's own prompt-end states count; states a shared prefix kept, which every
    conversation resumes from, are not touched."""

    value = (os.environ.get("TF_GLM_KEEP_PER_CHAT", "") if value is None else value).strip()
    if value == "":
        return 0
    if not value.isdecimal():
        raise ValueError(f"TF_GLM_KEEP_PER_CHAT: 0 (no limit) or a count, not {value!r}")
    return int(value)


KEEP_PER_CHAT = keep_per_chat()


def kept_bytes_cap() -> int:
    """TF_GLM_KEPT_BYTES_GIB: how much device memory the kept prompts own outside the pool, together (0, the default: no cap).

    A kept state holds tensors of its own besides its pool rows -- above all a copy of its DFlash2 window
    (``draft_window``), which every later block pass at its end reads. The entry cap alone does not bound them: with
    TF_GLM_CACHE_ENTRIES=64 and long conversations the copies reached tens of GiB on the hosts (their free memory fell
    to 0.1 GB), because nothing dropped them except the entry count and nothing returned the freed blocks to the
    driver. The single-stream path bounds the same memory by bytes (``engine._held_bytes`` against ``cache_bytes``);
    this is that bound for --parallel, where ``cache_bytes`` is 0 and the pool is assumed to hold everything."""

    value = os.environ.get("TF_GLM_KEPT_BYTES_GIB", "").strip()
    if not value:
        return 0
    gib = float(value)
    return 0 if gib <= 0 else int(gib * 2 ** 30)


KEPT_BYTES_CAP = kept_bytes_cap()


def known_key(c):
    """A kept state's key as an int64 array, converted once: every admission scans all of ``kept`` for the prefix a
    new prompt shares, and rebuilding the array each time cost O(kept states x their tokens) a request."""

    arr = getattr(c, "key_array", None)
    if arr is None:
        arr = c.key_array = np.asarray(c.key, dtype=np.int64)
    return arr


def cap_victim(kept: list):
    """The kept prompt to drop past TF_GLM_CACHE_ENTRIES (``kept``: least recently used first). An extent's kept
    prompts are prefixes of its rows, so one shorter than another in its extent is an earlier point of that
    conversation, superseded by the longer one: the oldest of those goes first, then the least recently used of any.
    A conversation taking many turns thus drops its own earlier turns instead of the only state of a conversation
    that is waiting on a long reply. A shared-prefix state (a system block, where prompts stop sharing) is never
    dropped as superseded: it is what the next conversation or run with that prefix resumes from, and its writer's
    longer state cannot serve them, so it goes by recency only, with the other conversations' latest states.
    Deterministic in ``kept``'s order and extents, so both ranks drop the same one."""

    longest: dict[int, int] = {}
    for c in kept:
        longest[id(c.extent)] = max(longest.get(id(c.extent), 0), len(c.ids))
    for c in kept:
        if len(c.ids) < longest[id(c.extent)] and not getattr(c, "shared", False):
            return c
    return kept[0]


def multi_code(code: list[int], dflash: bool, dflash_policy: list[int]) -> list[int]:
    """A request's policy code under --parallel: serial stays serial; auto becomes TF_GLM_DFLASH_POLICY; an MTP
    policy its DFlash2 twin; without a draft model, serial."""

    kind = code[0]
    if kind == 0 or not dflash:
        return [0, 0, 0, 0]
    if kind in (4, 5):
        return list(dflash_policy)
    return [kind + 10] + list(code[1:]) if kind < 10 else list(code)


def trim(drafts: list[list[int]], cap: int = MAX_WINDOW) -> list[list[int]]:
    """Cut drafts until every window (a pending token and its drafts) fits ``cap`` rows together: one draft at a
    time off the longest window (the latest of equals)."""

    drafts = [list(d) for d in drafts]
    total = sum(1 + len(d) for d in drafts)
    while total > cap:
        k = max(range(len(drafts)), key=lambda i: (len(drafts[i]), i))
        if not drafts[k]:
            break
        drafts[k].pop()
        total -= 1
    return drafts


def _f64(x: float) -> list[int]:
    import struct

    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _unf64(lo: int, hi: int) -> float:
    import struct

    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


def pack_sampling(sampling) -> list[int]:
    """10 int32 words: the seed (3 words of 31, 31 and 2 bits), temperature, top_k, top_p, min_p."""

    seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
    return [seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
            *_f64(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
            *_f64(sampling.top_p if sampling else 1.0), *_f64(sampling.min_p if sampling else 0.0)]


def unpack_sampling(w: Sequence[int]):
    from tensorfold.engine.exact_sampling import Sampling

    s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi = w
    temperature = _unf64(t_lo, t_hi)
    if temperature <= 0:
        return None
    return Sampling((s_top << 62) | (s_hi << 31) | s_lo, temperature, top_k, _unf64(p_lo, p_hi), _unf64(m_lo, m_hi))


# -- DFlash2 contexts per slot ---------------------------------------------------------------------------------------
def draft_window(ctx, n: int) -> list[torch.Tensor]:
    """A copy of a stream's DFlash2 window rows before n (``dflash2_multi.DraftContext``, or a solo ``Drafter``):
    what a later block pass at n or past it reads. A ring's are copied by ``take_snapshot`` already."""

    from .decode import _drafter_views, _ring_window

    return _ring_window(ctx, n) if getattr(ctx, "ring", 0) else [v.clone() for v in _drafter_views(ctx, n)]


def put_draft_window(ctx, n: int, rows: list[torch.Tensor]) -> None:
    """Put a kept prompt's DFlash2 window back and end the context at n."""

    from .decode import _drafter_views, _put_ring_window

    if getattr(ctx, "ring", 0):
        _put_ring_window(ctx, n, rows)
    else:
        views = _drafter_views(ctx, n)
        if len(views) != len(rows):
            raise ValueError("a kept state's DFlash2 window does not match the drafter's context")
        for dst, src in zip(views, rows):
            dst.copy_(src)
    ctx.context_end = n
    ctx.pos_dev.fill_(n)


# -- the verify seam -------------------------------------------------------------------------------------------------
def verify_kind(value: str | None = None) -> str:
    """TF_GLM_MULTI_VERIFY: how a round's windows are verified: batched (the default: one forward over every stream's
    rows, the segmented kernels) or serial (the reference: each stream's window as its own forward)."""

    value = (os.environ.get("TF_GLM_MULTI_VERIFY", "") if value is None else value).strip().lower() or "batched"
    if value not in ("batched", "serial"):
        raise ValueError(f"TF_GLM_MULTI_VERIFY: batched or serial, not {value!r}")
    return value


def forward_streams(verify, segments: Sequence[Segment]) -> Verified:
    """The seam: every segment's logits (and taps) of one round, by ``verify`` (``verify.SerialVerify`` or
    ``verify.BatchedVerify``, the same two calls)."""

    return verify.forward(segments)


def sample_streams(w, parts: Sequence[tuple]) -> list[list[int]]:
    """parts [(logits [R, V/world], positions, sampling)] -> each part's tokens, as ``decode.sample_rows`` samples
    each alone: every top-k (and greedy) part's candidates in ONE all-gather, a nucleus part (top_k off) through the
    shared nucleus rule (its own gathers), in part order on every rank."""

    from tensorfold.cuda.sampling import comm_gather, nucleus_rows, one_rank
    from tensorfold.engine.exact_sampling import MARGIN, choose_rows

    out: list[list[int] | None] = [None] * len(parts)
    packed, layout = [], []
    for k, (logits, positions, sampling) in enumerate(parts):
        greedy = sampling is None or sampling.temperature <= 0
        if not greedy and not sampling.top_k:
            continue
        n = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
        vals, ids = torch.topk(logits.float(), n, dim=-1)
        ids = (ids + w.vocab_offset).to(torch.int32)
        packed.append(torch.cat([vals, ids.view(torch.float32)], dim=1).reshape(-1))
        layout.append((k, logits.shape[0], n))
    if packed:
        flat = torch.cat(packed) if len(packed) > 1 else packed[0]
        if w.comm is None:
            host = flat.view(1, -1).cpu()
        else:
            got = torch.empty((w.world * flat.numel(),), dtype=torch.float32, device=flat.device)
            w.comm.all_gather(flat.contiguous(), got)
            host = got.view(w.world, -1).cpu()
        at = 0
        for k, R, n in layout:
            g = host[:, at:at + R * 2 * n].reshape(host.shape[0], R, 2 * n)
            at += R * 2 * n
            values = torch.cat([g[r, :, :n] for r in range(g.shape[0])], dim=1).numpy().astype(np.float32)
            tokens = torch.cat([g[r, :, n:].contiguous().view(torch.int32) for r in range(g.shape[0])],
                               dim=1).numpy().astype(np.int64)
            _, positions, sampling = parts[k]
            if sampling is None or sampling.temperature <= 0:
                order = np.lexsort((tokens, -values), axis=-1)
                out[k] = [int(tokens[i, order[i, 0]]) for i in range(R)]
            else:
                out[k] = choose_rows(values, tokens, positions, sampling)
    for k, (logits, positions, sampling) in enumerate(parts):
        if out[k] is None:
            out[k] = nucleus_rows(logits, positions, sampling, offset=w.vocab_offset,
                                  gather=one_rank if w.comm is None else comm_gather(w.comm))
    return out


def keyed_ids(prompt: Sequence[int], positions: Sequence[int], rows: Sequence[int], keys: Sequence[int]) -> list[int]:
    """The prompt's ids with each picture's rows replaced by -1 - its content key (``GlmPrepared.keyed_ids``): equal
    prefixes then mean the same pictures in the same places. Both ranks build them from the ADMIT's fields."""

    ids, at = [int(t) for t in prompt], 0
    for n, key in zip(rows, keys):
        for p in positions[at:at + n]:
            ids[p] = -1 - key
        at += n
    return ids


def pack_keys(rows: Sequence[int], keys: Sequence[int], positions: Sequence[int]) -> list[int]:
    """The ADMIT's picture fields: counts, rows a picture, each 62-bit key as two 31-bit halves, every picture row."""

    halves = [h for k in keys for h in (k >> 31, k & 0x7FFFFFFF)]
    return [len(rows), *rows, *halves, len(positions), *positions]


def unpack_keys(p: Sequence[int], i: int) -> tuple[list[int], list[int], list[int], int]:
    k = p[i]
    rows = list(p[i + 1:i + 1 + k])
    halves = p[i + 1 + k:i + 1 + 3 * k]
    keys = [(halves[2 * j] << 31) | halves[2 * j + 1] for j in range(k)]
    i += 1 + 3 * k
    m = p[i]
    return rows, keys, list(p[i + 1:i + 1 + m]), i + 1 + m



# -- streams -------------------------------------------------------------------------------------------------------------
@dataclass(eq=False)
class Lane:
    """An admitted stream: its request (``Stream``), slot, extent, state views and decoding state (both ranks)."""

    s: Stream
    sid: int
    slot: int
    extent: Any
    st: Any
    order: int
    code: list[int]
    spec: str = ""
    policy: Any = None                 # decode.DepthPolicy (DFlash2), None: one row a round
    dflash: bool = False               # the stream drafts with DFlash2 (its slot's context holds its taps)
    depth: int = 0
    copies: Any = None
    feed: Any = None
    constraint: Any = None
    window: Any = None                 # the grammar's window of this round's rows
    stops: list[int] = field(default_factory=list)   # prompt positions whose states are kept (before the end)
    point: int | None = None           # where the prompt's own state is kept (engine.grid_point), or None
    shared: list[int] = field(default_factory=list)
    head: Any = None                   # a replay's first-token logits row (its kept prompt's, ``decode.whole``)
    key: list[int] | None = None       # rank 0: the prompt's ids with its pictures' keys in their rows (kept states)
    images: bool = False               # both ranks: the prompt has pictures (its kept states are told apart by key)
    chat: int = 0                      # the conversation this stream belongs to (its kept states are capped together)
    decoding: bool = False
    paused: bool = False
    t0: float = 0.0
    copy_rounds: int = 0
    copy_drafted: int = 0
    copy_accepted: int = 0


class NoRoom(RuntimeError):
    """The pool cannot place a stream now (it waits for others to finish)."""


class MultiDecoder:
    """The ``GlmScheduler``'s decoder, on rank 0 (deciding) and rank 1 (``follow``, applying)."""

    window = MAX_WINDOW                 # rows of a round's windows together (TF_GLM_MULTI_WINDOW; set by __init__)
    kept_bytes_cap = 0                  # TF_GLM_KEPT_BYTES_GIB in bytes (0: off; set by __init__)

    def __init__(self, engine, streams: int, *, verify=None, drafts=None, draft_graphs: bool | None = None,
                 tune: MultiSettings | None = None, row_ms: Sequence[float] | None = None,
                 window: int | None = None) -> None:
        """``drafts``: the multi-stream drafter (``dflash2_multi.MultiDrafter``; default one over the engine's
        drafter, its graphs captured unless TF_GLM_MULTI_DRAFT_GRAPHS=0); ``verify``: the verify seam (default
        ``SerialVerify``); ``window``: the rows of a round's windows together (default the engine's
        TF_GLM_MULTI_WINDOW, ``engine.multi_window``, else MAX_WINDOW)."""
        from .engine import DFLASH_POLICY, encode_policy

        self.g = engine
        self.e = e = engine.e
        self.w = e.w
        self.rank = engine.rank
        self.kept_bytes_cap = KEPT_BYTES_CAP     # ours: what the kept states own outside the pool, together
        self.count = streams
        if e.slots.count < streams:
            raise ValueError(f"the engine holds {e.slots.count} stream slots, not {streams}")
        self.window = int(window if window is not None else getattr(engine, "multi_window", MAX_WINDOW))
        if self.window < streams:
            raise ValueError(f"a verify window of {self.window} rows cannot hold {streams} streams' pending tokens")
        self.rows_max = e.rows                           # a stream's window (decode.MAX_ROWS)
        self.pool = Pool(e.caches.rows)
        self.arena = e.caches.arena
        self.grid = engine.grid
        self.fill_rows = fill_rows(e.prefill_rows, self.grid)
        self.fill_budget = sliced_fill.budget_ms()
        self.fill_drafts = sliced_fill.drafted_rounds()     # rank 0 decides; the ROUND message carries it
        self.partial = None
        self.fill_costs = None
        self.decode_due = False
        self.short_round_due = False
        self.share = fill_share()
        self.credit = 0.0
        # TF_GLM_MULTI_PREFILL: rank 0 groups filling streams into one chunk (rank 1 applies whatever MFILL it gets)
        self.group = multi_prefill.enabled()
        if self.group:
            from . import latent

            if not latent.ENABLED:
                raise ValueError("TF_GLM_MULTI_PREFILL=1 needs the latent cache (TF_GLM_LATENT=1)")
        self.unit = fill_unit(self.grid)
        self.grouped = {"chunks": 0, "pieces": 0, "rows": 0}           # rank 0's multi-prompt chunks, for /health
        self.drafts = drafts
        if self.drafts is None and engine.drafter is not None:
            from .dflash2_multi import MultiDrafter

            self.drafts = MultiDrafter(engine.drafter, streams=streams)
            graphs = os.environ.get("TF_GLM_MULTI_DRAFT_GRAPHS", "1") != "0" if draft_graphs is None else draft_graphs
            if graphs and torch.cuda.is_available():
                self.drafts.capture()
        if verify is None:
            if verify_kind() == "batched":
                taps = engine.drafter.tap_layers if engine.drafter is not None else ()
                verify = BatchedVerify(e, taps=taps, rows=self.window)
                # TF_GLM_MULTI_GRAPHS=0: eager windows (the same bits); by default a graph a window size
                if os.environ.get("TF_GLM_MULTI_GRAPHS", "1") != "0" and torch.cuda.is_available():
                    verify.capture()
            else:
                verify = SerialVerify(e, taps=self.drafts is not None)
        self.verify = verify
        # speed settings (multi_tune): the sampler's gathers, the drafts' depth, async messages, the profile
        self.tune = tune if tune is not None else MultiSettings.from_env()
        self.profiles = None
        if self.tune.profile and self.rank == 0:
            self.profiles = {False: RoundProfile(self.tune.profile, label="batched"),
                             True: RoundProfile(self.tune.profile, label="lone stream, one-stream graphs")}
        self.profile = None                              # the current round's (``profiles``), or None
        self.solo_verify = SerialVerify(e, taps=self.drafts is not None)
        graphs = getattr(e, "graphs", None)
        self.lone_rows = max((r for r, _ in getattr(graphs, "main", {})), default=0)
        self.row_ms = list(row_ms) if row_ms is not None else None
        if self.tune.depth == "joint" and self.row_ms is None:
            self.row_ms = self._time_rows()
        # both ranks: the comparison replays graphs whose all-gathers the other rank must join (TF_GLM_MULTI_PROFILE
        # is in the two-rank settings comparison, so both run it or neither)
        if self.tune.profile and isinstance(self.verify, BatchedVerify) and self.verify.graphs and self.lone_rows:
            self._compare_paths()
        self.msg = None                                  # TF_GLM_MULTI_ASYNC: rank 0's pinned message staging
        self.watchdog = watchdog_seconds()               # stack dumps of a stalled iteration (TF_GLM_MULTI_WATCHDOG_S)
        watchdog_exits()                                  # TF_GLM_MULTI_WATCHDOG_EXIT read now: a typo fails at start
        self.iteration_since = None                       # when rank 0's current iteration began (/health)
        self.sent = self.received = 0                     # messages sealed on rank 0, unsealed on rank 1
        self.lanes: dict[int, Lane] = {}                 # by sid, in admission order
        self.kept: list = []                             # kept prompts (decode.Snapshot + kid, extent), oldest first
        self.disk = self._spill_store()                  # --spill-gib: the disk tier for kept prompts (cuda/spill.py)
        self.next_sid = self.next_kid = self.next_order = 0
        self.next_chat = 0
        self.outbox: list[int] = []
        self.idle = True
        self.broken: Exception | None = None
        self.requeue: list[Stream] = []                  # rank 0: streams given back to the scheduler's queue
        self.eos = tuple(self.w.cfg.eos)
        self.dflash_code = encode_policy(DFLASH_POLICY)
        if self.fill_budget:
            self.fill_costs = self._time_prompt_layers()

    # -- the scheduler's view ----------------------------------------------------------------------------------------
    @property
    def streams(self) -> dict[int, Stream]:
        """Decoding streams (``Scheduler._yield`` and /health read it)."""
        return {sid: lane.s for sid, lane in self.lanes.items() if lane.decoding}

    @property
    def filling(self) -> list[Stream]:
        return [lane.s for lane in self.lanes.values() if not lane.decoding]

    def live(self) -> int:
        return len(self.lanes)

    def health(self) -> dict:
        lanes = list(self.lanes.values())
        return {"streams": {"decoding": sum(l.decoding and not l.paused for l in lanes),
                            "filling": sum(not l.decoding for l in lanes),
                            "paused": sum(l.decoding and l.paused for l in lanes)},
                "pool_tokens": self.pool.rows, "pool_free_tokens": self.pool.free_rows(),
                "kept_prompts": len(self.kept),
                "kept_bytes": self._held_bytes(), "kept_bytes_cap": self.kept_bytes_cap,
                # what the slots are holding. own = a conversation's turn-boundary state (capped by
                # TF_GLM_KEEP_PER_CHAT); shared = a shared-prefix point or a mid-prefill pause state (not capped);
                # by_chat shows one conversation id per entry, so a long session that keeps forking its prefix
                # (compaction) and minting new ids is visible instead of guessed at.
                "kept_mix": {
                    "own": sum(1 for c in self.kept if getattr(c, "own", False)),
                    "shared": sum(1 for c in self.kept if not getattr(c, "own", False)),
                    "chats": len({getattr(c, "chat", 0) for c in self.kept}),
                    "rows": sum(len(c.ids) for c in self.kept),
                    "by_chat": sorted(([n, sum(1 for c in self.kept if getattr(c, "chat", 0) == n),
                                        max(len(c.ids) for c in self.kept if getattr(c, "chat", 0) == n)]
                                       for n in {getattr(c, "chat", 0) for c in self.kept}),
                                      key=lambda r: -r[1])[:8],
                },
                **({"multi_prefill": dict(self.grouped)} if self.group else {}),
                **({"spill": self.disk.info()} if self.disk is not None else {}),
                # how long rank 0's current iteration has run (0 between iterations): a stalled server keeps growing it
                "iteration_s": round(time.monotonic() - since, 1)
                if (since := getattr(self, "iteration_since", None)) is not None else 0.0}

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the two ranks are out of step after an error; restart both") from self.broken

    # -- messages --------------------------------------------------------------------------------------------------------
    def _emit(self, op: int, payload: Sequence[int]) -> list[int]:
        payload = [int(v) for v in payload]
        self.outbox += [op, len(payload), *payload]
        return payload

    def _flush(self) -> None:
        """Rank 0: this iteration's ops to rank 1, one message (after the doorbell when rank 1 idles)."""

        if self.idle:
            self.g._ring()
            self.idle = False
        msg, self.outbox = seal(self.outbox, self.sent), []
        self.sent = (self.sent + 1) % SEAL_MOD
        if self.tune.async_msg or (self.fill_budget and (self.partial is not None or self.short_round_due)):
            self._send(msg)
        else:
            self.g._share(msg)

    def _send(self, values: list[int]) -> None:
        """Rank 0, TF_GLM_MULTI_ASYNC: ``GlmEngine._share``'s two all-gathers (the length, then the values) without
        reading anything back: the values go through a pinned staging buffer (reused once its last copy has left),
        so the host does not wait for the GPU. Rank 1 receives them with its ``_share`` as ever. Each gather sends a
        device buffer of its own from its start, as rank 1's ``_share`` does: a transport may choose by the buffer's
        address (TF_GLM_COMM=roce did), and a view past an allocation's start once split the ranks' choices."""

        n = len(values)
        dev = self.w.device
        if self.msg is None or self.msg[0].numel() < n + 1:
            if self.msg is not None and self.msg[3] is not None:
                self.msg[3].synchronize()
            size = max(4096, 2 * (n + 1))
            pinned = torch.cuda.is_available()
            self.msg = (torch.zeros((size,), dtype=torch.int32, pin_memory=pinned),
                        torch.zeros((1,), dtype=torch.int32, device=dev),
                        torch.zeros((size,), dtype=torch.int32, device=dev),
                        torch.cuda.Event() if pinned else None)
        host, length, vals, ready = self.msg
        if ready is not None:
            ready.synchronize()                          # the last message's copies have left the staging buffer
        host[0] = n
        if n:
            host[1:1 + n].numpy()[:] = values
        length.copy_(host[:1], non_blocking=True)
        if n:
            vals[:n].copy_(host[1:1 + n], non_blocking=True)
        if ready is not None:
            ready.record()
        comm = self.g.comm
        world = self.g.world
        got = torch.empty((world,), dtype=torch.int32, device=dev)
        comm.all_gather(length, got)
        allv = torch.empty((world * n,), dtype=torch.int32, device=dev)
        comm.all_gather(vals[:n], allv)

    def _time_rows(self, reps: int = 5) -> list[float]:
        """TF_GLM_MULTI_DEPTH=joint: the batched window's ms for 1 .. its rows (its graphs, fastest of
        ``reps``), the larger of both ranks' (so both allocate alike)."""

        v = self.verify
        if not isinstance(v, BatchedVerify) or not v.graphs:
            raise ValueError("TF_GLM_MULTI_DEPTH=joint needs the batched verify window's graphs "
                             "(TF_GLM_MULTI_VERIFY=batched, TF_GLM_MULTI_GRAPHS=1)")
        mine = torch.tensor(v.time_rows(reps), dtype=torch.float32, device=self.w.device)
        world = self.g.world
        got = torch.empty((world * mine.numel(),), dtype=torch.float32, device=self.w.device)
        self.g.comm.all_gather(mine, got)
        return [float(x) for x in got.view(world, -1).max(dim=0).values.tolist()]

    @torch.no_grad()
    def _time_prompt_layers(self):
        """Both ranks calibrate isolated prompt forwards before requests are admitted (0 budget: no work)."""
        from .decode import prefill_chunk_steps
        from .verify import timing_tokens

        e = self.e
        samples = {}
        prev = e.use(e.home)
        try:
            # Warm kernels first; this never publishes a kept prompt or draft context.
            e.reset()
            for _ in prefill_chunk_steps(e, timing_tokens(self.w.cfg.vocab, 64), 0, 64, head=False):
                pass
            for rows in sorted({min(r, self.fill_rows) for r in (64, 256, 512, 1024)} | {self.fill_rows}):
                e.reset()
                for _ in prefill_chunk_steps(e, timing_tokens(self.w.cfg.vocab, rows), 0, rows, head=False):
                    pass                            # shape-specific warmup before its measured pass
                e.reset()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(len(self.w.layers) + 2)]
                steps = prefill_chunk_steps(e, timing_tokens(self.w.cfg.vocab, rows), 0, rows, head=False)
                events[0].record()
                for i, _ in enumerate(steps):
                    events[i + 1].record()
                events[-1].record()
                events[-1].synchronize()
                costs = [events[i].elapsed_time(events[i + 1]) for i in range(len(self.w.layers))]
                mine = torch.tensor(costs, dtype=torch.float32, device=self.w.device)
                if self.w.comm is not None:
                    got = torch.empty((self.w.world * mine.numel(),), dtype=mine.dtype, device=mine.device)
                    self.w.comm.all_gather(mine, got)
                    costs = got.view(self.w.world, -1).max(dim=0).values.cpu().tolist()
                samples[rows] = costs
        finally:
            e.reset()
            e.last_hidden = None
            e.use(prev)
        return sliced_fill.LayerCosts(samples)

    def _compare_paths(self, reps: int = 5) -> None:
        """TF_GLM_MULTI_PROFILE (at startup, on both ranks: the graphs all-gather; rank 0 prints): a lone stream's
        verify window on the one-stream graphs and on the batched graphs, ms by rows up to the one-stream graphs'
        widest (fastest of ``reps``; slot 0's states cleared after)."""

        from .forward import stage
        from .verify import timing_tokens

        e, v = self.e, self.verify
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        solo = v.time_rows(reps, most=self.lone_rows)
        tokens = timing_tokens(self.w.cfg.vocab, v.rows_max)       # the rows time_rows gave the batched graphs
        out = []
        for R in range(1, self.lone_rows + 1):
            g = e.graphs.main.get((R, e.home.parity))
            if g is None:
                continue
            stage(self.w, e.home, e.buf, tokens[:R])
            best = float("inf")
            for _ in range(reps + 1):
                start.record()
                g.replay()
                stop.record()
                stop.synchronize()
                best = min(best, start.elapsed_time(stop))
            out.append(f"{R}: {best:.2f} vs {solo[R - 1]:.2f}")
        e.slots.rec[0].zero_()
        e.slots.conv[0].zero_()
        if self.rank == 0:
            print("[tensorfold] multi profile: a lone stream's verify window ms, one-stream graphs vs batched graphs, by "
                  "rows: " + ", ".join(out), flush=True)

    def _row_cost(self, rows: int) -> float:
        ms = self.row_ms
        return ms[min(max(rows, 1), len(ms)) - 1] + max(0, rows - len(ms)) * (ms[-1] - ms[-2] if len(ms) > 1 else 0)

    @staticmethod
    def parse(msg: Sequence[int]) -> list[tuple[int, list[int]]]:
        ops, i = [], 0
        while i < len(msg):
            if i + 2 > len(msg):
                raise RuntimeError("truncated multi-stream op header")
            op, n = int(msg[i]), int(msg[i + 1])
            if op not in OPS or n < 0 or i + 2 + n > len(msg):
                raise RuntimeError("invalid multi-stream op or payload length")
            p = [int(v) for v in msg[i + 2:i + 2 + n]]
            if op == FILLPART and (len(p) != 4 or not 0 <= p[2] < p[3]):
                raise RuntimeError("invalid FILLPART payload")
            if op == MFILLPART and (len(p) < 5 or p[0] < 1 or len(p) != 3 + 2 * p[0]
                                    or not 0 <= p[1] < p[2]):
                raise RuntimeError("invalid MFILLPART payload")
            ops.append((op, p))
            i += 2 + n
        return ops

    # -- admission (rank 0 decides) ---------------------------------------------------------------------------------------
    def _need(self, prompt: int) -> int:
        return align_up(prompt + self.rows_max)

    def fits(self, s: Stream) -> bool:
        """Whether ``s`` can be placed now (else it waits, in order, for streams to finish): the free rows and the
        kept-only extents cover its prompt, with a block of headroom for each decoding stream to grow into."""

        if not self.lanes:
            return True                  # everything can be evicted; a prompt past the pool fails in ``admit``
        need = self._need(len(s.prompt))
        room = self.pool.free_rows() + sum(x.size for x in self.pool.extents if x.owner is None)
        return room >= need + ALIGN * sum(l.decoding for l in self.lanes.values())

    def admit(self, s: Stream) -> None:
        """Rank 0: place a request (slot, extent; resumed from the longest kept prefix of its prompt when it drafts)
        and queue its ADMIT; its prompt is prefilled chunk by chunk in later iterations."""

        from .engine import grid_point, shared_points

        self._check()
        g = self.g
        prompt = [int(t) for t in s.prompt]
        if not prompt:
            raise ValueError("a request needs at least one prompt token")
        if len(prompt) >= g.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {g.limit}")
        s.count = max(1, min(int(s.count), g.limit - len(prompt)))
        if self._need(len(prompt)) > self.pool.rows:
            raise ValueError(f"a prompt of {len(prompt)} tokens does not fit the {self.pool.rows}-token pool")
        info = getattr(s, "glm", None) or {}
        draft = bool(s.draft) and not g.serial_only
        code = multi_code(info.get("code") or [0, 0, 0, 0], self.drafts is not None and draft,
                          self.dflash_code) if draft else [0, 0, 0, 0]
        positions: list[int] = []
        feed = None
        key = prompt
        vrows: list[int] = []
        vkeys: list[int] = []
        vpos: list[int] = []
        if s.vision is not None:
            if g.vision is None:
                raise ValueError("image inputs require starting this server with --vision")
            positions = [int(p) for p in s.vision.positions]
            if positions and (positions[-1] >= len(prompt) or positions != sorted(set(positions))):
                raise ValueError("image rows must sit at increasing positions inside the prompt")
            keyed = getattr(s.vision, "item_keys", None)
            if keyed and sum(s.vision.item_rows) == len(positions):
                vrows, vkeys, vpos = [int(r) for r in s.vision.item_rows], [int(k) for k in keyed], list(positions)
                key = keyed_ids(prompt, vpos, vrows, vkeys)
            else:
                key = None
        slot = next((k for k in range(self.count) if k not in {l.slot for l in self.lanes.values()}), None)
        if slot is None:
            raise NoRoom(f"every one of the {self.count} stream slots is taken")
        hit = self._resume(key) if draft and key is not None else None
        encoded_from = len(hit.ids) if hit is not None else 0
        if s.vision is not None:              # encoded before anything is sent: a failure stops here
            feed = self._encode(s.vision, encoded_from)
        need = self._need(len(prompt))
        # placement: take a kept extent over in place, or copy its rows into a new one while a stream writes in it
        copy = 0
        base = None
        if hit is not None and hit.extent.owner is None:
            x = hit.extent
            # kept prompts the stream would overwrite: another conversation's longer states in the same rows (a shared
            # system prompt's state lives in the extent of the conversation that wrote it). Copy the shared rows into
            # an extent of their own instead, so that conversation keeps its state. Only into free rows: evicting other
            # conversations to make room would cost more than it saves, so then the rows are overwritten as before.
            # Not with TF_GLM_MULTI_LONE=1: the copy, decoding alone, would move home to the pool's first rows and evict
            # the writer's whole extent there, the shared state included.
            others = [c for c in x.kept if len(c.ids) > len(hit.ids) and key[:len(c.ids)] != c.key]
            if others and not self.tune.lone:
                base = self.pool.place(need)
                copy = int(base is not None)
            if base is None:
                for c in others:
                    self._evict(c)
                if self._grow(x, need, protect=[x]):
                    base = x.base
                else:
                    hit = None
        if base is None:
            base = self._room(need, [hit.extent] if hit is not None else [])
            if base is None and hit is not None:          # no room beside the kept rows: start fresh
                hit = None
                base = self._room(need)
            if base is None:
                raise NoRoom(f"no room for a {need}-token extent in the {self.pool.rows}-token pool now")
            copy = int(hit is not None)
        if s.vision is not None and hit is None and encoded_from:
            feed = None                       # (its rows freed before the tower runs again)
            feed = self._encode(s.vision, 0)  # the kept state was given up: every picture, from the prompt's start
        positions = feed.positions if feed is not None else []
        cut = len(hit.ids) if hit is not None else 0
        shared: list[int] = []
        if draft and key is not None and g.shared:
            known = [known_key(c) for c in self.kept if c.key is not None]
            shared = shared_points(key, cut, grid_point(len(prompt), cut, self.grid), self.grid, g.shared, known,
                                   g.opener)
        sid = self.next_sid
        payload = [sid, slot, base, need, hit.kid if hit is not None else -1, copy, s.count, int(s.stop_eos),
                   int(draft), *pack_sampling(s.sampling), *(shared + [0] * SHARED_SLOTS)[:SHARED_SLOTS], *code,
                   len(prompt), *prompt]
        from tensorfold.engine.grammar import pack

        packed = pack(s.constraint)
        payload += [len(packed), *packed, len(positions), *positions, int(s.vision is not None),
                    *pack_keys(vrows, vkeys, vpos)]
        self._emit(ADMIT, payload)
        try:
            self._admitted(self._parse_admit(payload), s, feed, s.constraint, info.get("spec", ""))
        except Exception as exc:              # rank 1 applies the same ADMIT: the ranks may no longer agree
            self.broken = exc
            raise
        self.lanes[sid].key = key

    def _encode(self, prepared, cut: int):
        """Rank 0: the tower's rows of the pictures a prompt resumed at ``cut`` still fills (those whose rows end past
        it), as a VisionFeed; None when the kept state already holds them all."""

        from .engine import VisionFeed

        first, end = 0, 0
        if cut and getattr(prepared, "item_rows", None):
            for rows in prepared.item_rows:
                if prepared.positions[end + rows - 1] >= cut:
                    break
                end += rows
                first += 1
            if first == len(prepared.item_rows):
                return None
            prepared = prepared.from_item(first)
        positions = [int(p) for p in prepared.positions]
        if not positions:
            return None
        t = time.perf_counter()
        torch.cuda.empty_cache()
        rows = self.g.vision.features(prepared)
        torch.cuda.empty_cache()
        if rows.shape[0] != len(positions):
            raise ValueError(f"the tower gave {rows.shape[0]} rows for {len(positions)} image positions")
        feed = VisionFeed(self.g, positions, rows)
        feed.encode_s = time.perf_counter() - t
        return feed

    @staticmethod
    def _parse_admit(p: list[int]) -> dict:
        (sid, slot, base, size, kid, copy, count, stop_eos, draft) = p[:9]
        sampling = unpack_sampling(p[9:19])
        shared = [v for v in p[19:19 + SHARED_SLOTS] if v]
        i = 19 + SHARED_SLOTS
        code = p[i:i + 4]
        i += 4
        n = p[i]
        prompt = p[i + 1:i + 1 + n]
        i += 1 + n
        n = p[i]
        packed = p[i + 1:i + 1 + n]
        i += 1 + n
        n = p[i]
        positions = p[i + 1:i + 1 + n]
        images = bool(p[i + 1 + n])
        vrows, vkeys, vpos, _ = unpack_keys(p, i + 2 + n)
        keyed = keyed_ids(prompt, vpos, vrows, vkeys) if vrows else None
        return dict(sid=sid, slot=slot, base=base, size=size, kid=kid, copy=copy, count=count, stop_eos=bool(stop_eos),
                    draft=bool(draft), sampling=sampling, shared=shared, code=code, prompt=prompt, packed=packed,
                    positions=positions, images=images, keyed=keyed)

    def _admitted(self, a: dict, s: Stream, feed, constraint, spec: str = "") -> None:
        """Both ranks: an ADMIT's stream in its slot and extent, resumed from its kept prompt if it names one."""

        from .engine import decode_policy, grid_point
        from .forward import State

        g, e = self.g, self.e
        hit = self._kept_by_id(a["kid"]) if a["kid"] >= 0 else None
        prompt = a["prompt"]
        s.prompt = list(prompt)
        cut = len(hit.ids) if hit is not None else 0
        if hit is not None and hit.ids != prompt[:cut]:
            raise RuntimeError(f"the kept prompt {a['kid']} is not a prefix of stream {a['sid']}'s prompt")
        if hit is not None and not a["copy"]:
            x = hit.extent
            if (x.base, x.size) != (a["base"], a["size"]) or x.owner is not None:
                raise RuntimeError(f"stream {a['sid']} takes over extent {x.eid} at [{x.base}, {x.end}), "
                                   f"rank 0 said [{a['base']}, {a['base'] + a['size']})")
            self._rewrites(x, cut)
        else:
            x = self.pool.add(a["base"], a["size"])
            if hit is not None:
                self.arena.copy(hit.extent.base, x.base, cut)
        x.owner = a["sid"]
        st = State(self.w, x.size, e.rows, caches=e.caches, base=x.base, slots=e.slots, slot=a["slot"])
        code = a["code"]
        dflash = code[0] >= 10 and self.drafts is not None
        policy = decode_policy(code, g.costs) if code[0] else None
        # the conversation this stream belongs to: the chain it resumed from when that state is a turn's own, else a
        # new one (a shared-prefix state belongs to everyone, so resuming from it starts no chain)
        if hit is not None and getattr(hit, "own", False) and getattr(hit, "chat", 0):
            chat = hit.chat
        else:
            self.next_chat += 1
            chat = self.next_chat
        lane = Lane(s, a["sid"], a["slot"], x, st, self.next_order, code, spec, policy, dflash, feed=feed,
                    constraint=constraint, images=a["images"], chat=chat)
        if a.get("keyed") is not None:              # rank 1 too: its kept states are stored and matched by them
            lane.key = a["keyed"]
        self.next_sid = max(self.next_sid, a["sid"] + 1)
        self.next_order += 1
        s.sid, s.cached = a["sid"], cut
        # the slot's state: the kept prompt's, or fresh
        st.reset()
        if self.drafts is not None:
            self._ctx(lane).reset()
        if hit is not None:
            from .decode import put_ring_tail

            st.rec[st.cur[0]].copy_(hit.rec)
            st.conv.copy_(hit.conv)
            put_ring_tail(st, hit)          # the index rows of its pools still filling, into this slot's rings
            st.set_pos(cut)
            if dflash:
                if hit.drafter_end != cut or hit.drafter_rows is None:
                    raise RuntimeError("a kept prompt without DFlash2's window was chosen for a DFlash2 stream")
                put_draft_window(self._ctx(lane), cut, hit.drafter_rows)
            if cut == len(prompt):          # a replay: its one FILL samples the kept head's row, prefilling nothing
                if hit.head is None:
                    raise RuntimeError(f"the kept prompt {a['kid']} is the whole prompt of stream {a['sid']} but "
                                       f"kept no head row")
                lane.head = hit.head.clone()
            self._touch(hit)
        # where its prompt states are kept: the grid point (or the end) and rank 0's shared-prefix points
        if a["draft"]:
            lane.point = grid_point(len(prompt), cut, self.grid)
            lane.shared = sorted(p for p in set(a["shared"]) if lane.point is not None and cut < p < lane.point)
            lane.stops = list(lane.shared) + ([lane.point] if lane.point is not None and lane.point < len(prompt)
                                              else [])
        s.started = time.perf_counter()
        lane.t0 = s.started
        self.lanes[lane.sid] = lane

    def _ctx(self, lane: Lane):
        """The lane's slot's DFlash2 context."""
        return self.drafts.contexts[lane.slot]

    # -- kept prompts ------------------------------------------------------------------------------------------------------
    def _kept_by_id(self, kid: int):
        for c in self.kept:
            if c.kid == kid:
                return c
        raise RuntimeError(f"rank {self.rank} holds no kept prompt {kid}")

    def _resume(self, key: list[int]):
        """The longest kept strict prefix of ``key`` (a prompt's keyed ids, ``Lane.key``), or the whole prompt when it
        kept its head's logits row (a replay of an earlier prompt), whose DFlash2 window came along."""
        best = None
        for c in self.kept:
            n = len(c.ids)
            fits = c.drafter_end == n and c.drafter_rows is not None or self.drafts is None
            fits = fits and not (self.grid and n % self.grid)
            short = n < len(key) or (n == len(key) and c.head is not None)
            if fits and short and c.key is not None and key[:n] == c.key and (best is None or n > len(best.ids)):
                best = c
        return best

    def _touch(self, c) -> None:
        self.kept = [k for k in self.kept if k is not c] + [c]

    def _keep(self, lane: Lane, n: int, head=None, own: bool = False) -> None:
        """Both ranks: the lane's state at prompt position n (== its pos) becomes a kept prompt in its extent;
        ``head``: at the prompt's end, the logits row its first token is sampled from (kept for a replay);
        ``own``: the prompt's own end state -- its conversation's turn boundary, capped by TF_GLM_KEEP_PER_CHAT --
        rather than a shared-prefix state another conversation resumes from."""

        from .decode import take_snapshot

        prev = self.e.use(lane.st)
        try:
            snap = take_snapshot(self.e, lane.s.prompt[:n], None, mtp=False, drafter=None)
        finally:
            self.e.use(prev)
        snap.drafter_end = -1
        if lane.dflash:
            snap.drafter_end = n
            snap.drafter_rows = draft_window(self._ctx(lane), n)
        snap.head = head
        # rank 0's key: a text prompt's is its ids (shared, not a second list); rank 1 keeps none
        snap.key = None if lane.key is None else snap.ids if not lane.images else lane.key[:n]
        snap.images = lane.images
        snap.kid = self.next_kid
        self.next_kid += 1
        snap.extent = lane.extent
        snap.shared = n in lane.shared
        snap.own, snap.chat = bool(own), int(lane.chat)
        # the same ids: the same prompt, unless pictures are in it (rank 1 has no keys; rank 0 resumes by key)
        for c in [c for c in self.kept if c.ids == snap.ids and not (c.images or snap.images)]:
            self._drop(c, spill=False)
        lane.extent.kept.append(snap)
        self.kept.append(snap)
        if own and KEEP_PER_CHAT:
            # this conversation's older turn states: a replay of a prompt that far back is all they serve, and holding
            # every one of them is how one long conversation fills the whole cache by itself. Least recently used
            # first (self.kept's order, a resumed state moved last), so an edited or retried branch -- whose states
            # are shorter than the line it left -- keeps its own. Superseded turns are not worth the disk tier:
            # dropped without a spill (the same on both ranks).
            mine = [c for c in self.kept if getattr(c, "own", False) and getattr(c, "chat", 0) == snap.chat]
            for c in mine[:-KEEP_PER_CHAT]:
                self._drop(c, spill=False)
        while len(self.kept) > max(1, self.g.cache_entries):
            self._drop(cap_victim(self.kept))
        # The entry cap is not a memory budget. Long conversations make each kept state own hundreds of MB outside
        # the pool (copies of the DFlash2 window), so 64 entries reached tens of GiB on the hosts (free memory to
        # 0.1 GB) and never came back. With TF_GLM_KEPT_BYTES_GIB set, rank 0 drops states past it, by the cap's
        # victim order, and tells rank 1 with EVICT (the same rank-0-decides path the takeover uses); the freed blocks
        # go back to the driver in _drop. A state dropped for its bytes is not written to the spill tier.
        if self.rank == 0 and self.kept_bytes_cap:
            while len(self.kept) > 1 and self._held_bytes() > self.kept_bytes_cap:
                victim = cap_victim(self.kept)
                if victim is None:
                    break
                self._evict(victim, spill=False)

    def _release(self) -> None:
        """Hand cached-but-unused blocks back to the driver, but only when torch is actually hoarding them (with
        TF_GLM_KEPT_BYTES_GIB set).

        `engine._remember`/`_take_over` call empty_cache after dropping states, and the parallel path never did -- the
        free memory the hosts lost (down to 0.1 GB) never came back. Calling it on every drop is too blunt though:
        measured by the author, that cost ~16% of decode speed (50.5 against 60.4 tok/s), because every round then had
        to ask the driver for its blocks again. So pay it only when the allocator holds a lot of unused memory."""

        try:
            if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < (512 << 20):
                return
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _held_bytes(self) -> int:
        """What every kept prompt owns outside the pool (its KDA states, conv windows, a kept DFlash2 window, the
        indexer rings' tail, a kept head row and any saved rows) -- the memory the entry cap does not bound."""

        from .decode import snapshot_bytes

        return sum(snapshot_bytes(c) for c in self.kept)

    def _drop(self, c, spill: bool = True) -> None:
        """Forget a kept prompt; its extent goes when nothing else holds it, else shrinks to what is kept. With the
        spill tier on it goes to disk first (its rows are still in place), unless ``spill`` is False or it is there."""

        if spill and self.disk is not None:
            self._spill(c, leaving=True)
        held = 0
        if self.kept_bytes_cap:
            from .decode import snapshot_bytes

            held = snapshot_bytes(c)
        self.kept = [k for k in self.kept if k is not c]
        x = c.extent
        x.kept = [k for k in x.kept if k is not c]
        c.rec = c.conv = None
        c.tail = c.head = None
        c.drafter_rows = None
        self._settle(x)
        if held >= (64 << 20):
            self._release()

    def _settle(self, x) -> None:
        if x.owner is not None:
            return
        if not x.kept:
            self.pool.remove(x)
            return
        size = align_up(max(len(k.ids) for k in x.kept))
        if size < x.size:
            self.pool.resize(x, size)

    def _evict(self, c, spill: bool = True) -> None:
        """Rank 0: drop a kept prompt on both ranks."""

        self._emit(EVICT, [c.kid, int(spill)])
        self._drop(c, spill=spill)

    # -- the spill tier (``tensorfold serve --spill-gib``; cuda/spill.py) ----------------------------------------------
    # Every call below runs on both ranks in the same order (rank 0 decides, rank 1 follows its ops); rank 0 sends what
    # it has queued before each of the tier's all-gathers (``_sync``). Rows a pending write reads, or a read is filling,
    # never move or go before the copy ends: the pool's remove / move / resize are wrapped (``_guard_pool``). A kept
    # prompt's fields go through ``spill.encode``; its pool rows go as named row tensors into a new extent's rows.
    _SPILL_SKIP = ("ids", "rows", "nbytes", "pending", "mtp_len", "kid", "extent", "key", "images", "shared",
                   "spill_key", "superseded", "own", "chat")

    def _spill_store(self):
        cfg = getattr(self.g, "spill_cfg", None)
        if cfg is None:
            return None
        from tensorfold.cuda.spill import SpillStore, build_id, weights_id

        for p in self.arena.planes:
            if ALIGN % p.div:
                raise ValueError(f"--spill-gib: a cache plane of {p.div} tokens a row does not divide {ALIGN}")
        planes = [(tuple(p.tensor.shape[1:]), str(p.tensor.dtype), p.div, p.pad) for p in self.arena.planes]
        drafter = getattr(self.g, "drafter", None)
        model = os.path.basename(str(getattr(self.g, "model_dir", "")).rstrip("/"))
        # what the stored bits depend on: build and runtime, drafter, KV and dense formats, TP layout, plane shapes,
        # grid, and the settings every rank agreed at start
        signature = "|".join(["multi", build_id(self.g.model_dir, self.g.drafter_dir),
                              "" if drafter is None else type(drafter).__name__,
                              os.environ.get("TF_GLM_KV", ""), os.environ.get("TF_GLM_DENSE", ""),
                              f"tp{self.w.world}", repr(planes), str(self.grid), str(self.drafts is not None),
                              repr(self.g.agreed)])
        self.spill_jobs: list = []             # (job, extent, rows it reads up to), both ranks alike
        self.loads: dict = {}                  # load id -> a stored prompt being read into a new extent
        self.next_lid = 0
        store = SpillStore(cfg, rank=self.rank, world=self.w.world, device=self.w.device, signature=signature,
                           weights=weights_id(self.g.model_dir, self.g.drafter_dir),
                           classes=["tensorfold.families.glm5_next.cuda.decode:Snapshot"],
                           model_id=model, share=self.g._share, gather=self.g._gather_ints,
                           spanned=[f"p{i}" for i, p in enumerate(self.arena.planes) if p.spanned])
        self._guard_pool()
        return store

    def _sync(self) -> None:
        """Rank 0: what is queued goes to rank 1 now (before one of the tier's all-gathers)."""

        if self.rank == 0 and self.outbox:
            self._flush()

    def _guard_pool(self) -> None:
        pool = self.pool
        remove, move, resize = pool.remove, pool.move, pool.resize

        def g_remove(x):
            self._release_rows(x, 0)
            return remove(x)

        def g_move(x, base, size=None):
            self._release_rows(x, 0)
            return move(x, base, size)

        def g_resize(x, size):
            self._release_rows(x, size)
            return resize(x, size)

        pool.remove, pool.move, pool.resize = g_remove, g_move, g_resize

    def _release_rows(self, x, keep: int) -> None:
        """Both ranks, each on its own: rows of extent x past ``keep`` are about to move or go; a pending write that
        reads them first finishes its copy off the device, a read filling them first ends (no message: every rank
        waits for its own)."""

        h = getattr(x, "loading", None)
        if h is not None and h.lj is not None:
            h.lj.done.wait()
        if self.disk is None or not self.spill_jobs:
            return
        self.spill_jobs = [j for j in self.spill_jobs if not j[0].copied.is_set()]
        for job, y, upto in self.spill_jobs:
            if y is x and upto > keep:
                self.disk.release(job)

    def _rewrites(self, x, cut: int) -> None:
        """Both ranks: a stream takes extent x over and writes its rows from ``cut`` on. A longer kept prompt there,
        evicted for it (``admit``), may still be on its way to disk: its copy off the device ends first."""

        div = max(p.div for p in self.arena.planes)
        self._release_rows(x, cut - cut % div)

    def _planes(self, base: int, n: int) -> list:
        out = []
        for i, p in enumerate(self.arena.planes):
            a = base // p.div
            out.append((f"p{i}", p.tensor[a:a + -(-n // p.div)]))
        return out

    def _key(self, c) -> tuple[int, int]:
        """Kept prompt ``c``'s ``ids_key``, hashed once (its ids never change while kept; ``_behind`` asks every
        iteration)."""

        k = getattr(c, "spill_key", None)
        if k is None:
            from tensorfold.cuda.spill import ids_key
            k = c.spill_key = ids_key(self._ident(c))
        return k

    @staticmethod
    def _ident(c) -> list[int]:
        """What the tier stores and matches a kept prompt by: its ids, or its keyed ids when it has pictures."""

        return c.key if c.images else c.ids

    def _superseded(self, c) -> bool:
        """A longer kept prompt of the same conversation (the same extent, it extends c) that can be written itself:
        c need not be. Never a shared-prefix state (other conversations resume from it, not from the longer one). The
        answer holds while the extent keeps the same prompts, so it is remembered against their kids."""

        x = getattr(c, "extent", None)
        if x is None or getattr(c, "shared", False):
            return False
        held = tuple(k.kid for k in x.kept)
        memo = getattr(c, "superseded", None)
        if memo is not None and memo[0] == held:
            return memo[1]
        n = len(c.ids)
        out = any(k is not c and len(k.ids) > n and k.ids[:n] == c.ids and self._spillable(k) for k in x.kept)
        c.superseded = (held, out)
        return out

    def _unspillable(self, c) -> str | None:
        """Why kept prompt ``c`` cannot be written (None: it can)."""

        if c.images and c.key is None:
            return "images"                     # pictures without content keys: nothing tells them apart
        n = len(c.ids)
        if n < self.disk.cfg.min_tokens:
            return "short"
        if self.grid and n % self.grid:
            return "off_grid"
        if c.rec is None:
            return "no_state"
        if self.drafts is not None and not (c.drafter_end == n and c.drafter_rows is not None):
            return "no_draft_window"
        return None

    def _spillable(self, c) -> bool:
        return self._unspillable(c) is None

    def _unwritten(self, why: str) -> None:
        st = self.disk.stats
        st.unwritten_evicted += 1
        st.unwritten[why] = st.unwritten.get(why, 0) + 1

    def _spill(self, c, leaving: bool = False, keep: frozenset = frozenset()) -> bool:
        """Both ranks, in lockstep: write kept prompt ``c`` (its rows still in place) to the spill tier, unless it is
        there or on its way, superseded, off grid, too short, or lacks the DFlash2 window a stream resuming from it
        needs. ``leaving``: its rows go right after (an eviction). ``keep``: stored entries no room is made from. True
        when a write was queued."""

        d = self.disk
        why = self._unspillable(c) or ("superseded" if self._superseded(c) else None)
        if why is not None:
            if leaving:
                self._unwritten(why)
            return False
        if d.has(self._ident(c), self._key(c)):
            return False
        self._sync()
        try:
            job = d.save(self._ident(c), c, self._planes(c.extent.base, len(c.ids)), skip=self._SPILL_SKIP,
                         head=c.head is not None, tokens=c.ids if c.images else None, keep=keep)
        except Exception as exc:                 # noqa: BLE001  (a spill never fails the engine: the prompt is lost)
            print(f"[tensorfold] spill rank {self.rank}: not written: {exc!r}", flush=True)
            job = None
        if job is None:
            if leaving:
                self._unwritten("not_saved")
            return False
        self.spill_jobs.append((job, c.extent, len(c.ids)))   # its rows stay put until copied (``_guard_pool``)
        return True

    def _kept_keys(self) -> frozenset:
        """Both ranks alike: the stored copies an early write never evicts, those of the prompts still kept (else
        they evict each other in turn under a small cap)."""

        return frozenset(self._key(k) for k in self.kept if self._spillable(k))

    def _behind(self) -> None:
        """Rank 0: past the pool's high-water mark (``--spill-highwater``), or near the kept-prompt cap, the kept
        prompts eviction would take next are written in the background (SPILL), so an eviction usually finds them on
        disk and frees their rows at once. A writer's worth at a time; more wait for the next iteration (deferred)."""

        d = self.disk
        rows = self.pool.rows
        used = rows - self.pool.free_rows()
        hw = d.cfg.highwater
        near_cap = len(self.kept) >= max(1, self.g.cache_entries) - 2
        if used < hw * rows and not near_cap:
            return
        token = (d.generation, tuple(k.kid for k in self.kept))  # what decides whether an early write finds room
        if getattr(self, "behind_skip", (None,))[0] != token:
            self.behind_skip = (token, set())
        skip = self.behind_skip[1]               # kept prompts that found no room: not tried again until then
        keep = None
        order = self._victims(())
        if near_cap and self.kept:
            v = cap_victim(self.kept)
            order = [v] + [c for c in order if c is not v]
        want = (used - hw * rows) + 2 * (1.0 - hw) * rows
        covered = 0
        for c in order:
            if covered >= want and not (near_cap and c is order[0]):
                break
            if (c.images and c.key is None) or c.kid in skip:    # (pictures without content keys: never written)
                continue
            if d.has(self._ident(c), self._key(c)) or self._superseded(c):
                covered += len(c.ids)
                continue
            if not self._spillable(c):
                continue
            if d.pending_jobs() >= d.cfg.writers:
                d.stats.deferred += 1
                break
            self._emit(SPILL, [c.kid])
            keep = self._kept_keys() if keep is None else keep
            refused = d.stats.full + d.stats.no_space
            if self._spill(c, keep=keep):
                d.stats.behind += 1
            elif d.stats.full + d.stats.no_space > refused:   # no room under the cap or above the free-disk floor
                skip.add(c.kid)
                continue
            covered += len(c.ids)

    # reads: rank 0 sends LOAD when a request's prompt extends a stored prompt longer than any kept one; both ranks
    # add an extent for it and start their reads (other streams keep decoding meanwhile); when rank 0's read is done
    # it sends LOADED and both agree, in one all-gather, that every read worked: the extent then holds a kept prompt
    # the request resumes from; else it goes, the entry is forgotten, and the request prefills.
    def prestage(self, s: Stream):
        """Rank 0, before a request is admitted: a read of the stored prompt its prompt extends, or None (nothing to
        read, or no room). The scheduler admits the request once ``load_ready``, after ``complete_load``."""

        if self.disk is None or not s.draft:
            return None
        self._check()
        prompt = [int(t) for t in s.prompt]
        if s.vision is not None:                 # pictures: matched by content (keyed ids), or not at all
            v = s.vision
            keys = getattr(v, "item_keys", None)
            if not keys or sum(v.item_rows) != len(v.positions):
                return None
            prompt = keyed_ids(prompt, [int(p) for p in v.positions], [int(r) for r in v.item_rows],
                               [int(k) for k in keys])
        if self._need(len(prompt)) > self.pool.rows:
            return None
        for h in self.loads.values():            # the same stored prompt on its way already: wait for that read
            if len(prompt) >= h.n and prompt[:h.n] == h.ids:
                return h
        hit = self._resume(prompt)
        e = self.disk.find(prompt, grid=self.grid)
        if e is None or (hit is not None and len(hit.ids) >= e.n):
            return None
        size = align_up(e.n)
        self._emit(TOUCH, list(e.key))           # both ranks: the newest entry now, so no save made for room evicts it
        self.disk.touch_key(e.key)
        base = self._room(size, [hit.extent] if hit is not None else [])
        if base is None:
            self._sync()
            return None
        hi, lo = e.key
        lid = self.next_lid
        self._emit(LOAD, [hi, lo, e.n, base, size, lid])
        self._flush()
        h = self._start_load(hi, lo, e.n, base, size, lid)
        h.ids = [int(t) for t in e.ids]
        return h

    def _start_load(self, hi: int, lo: int, n: int, base: int, size: int, lid: int):
        """Both ranks: an extent at ``base`` for stored prompt (hi, lo), its read started on the reader thread."""

        import types

        self.next_lid = lid + 1
        x = self.pool.add(base, size)
        h = types.SimpleNamespace(lid=lid, key=(hi, lo), n=n, x=x, lj=None, err=None, done=False, ids=None)
        x.loading = h

        def dst(name, shape, dt):
            if name.startswith("rows.p"):
                p = self.arena.planes[int(name[6:])]
                a = base // p.div
                return p.tensor[a:a + shape[0]]
            return None

        try:
            h.lj = self.disk.load_async((hi, lo), dst)
        except Exception as exc:                 # noqa: BLE001  (the read fails at LOADED, on every rank)
            h.err = exc
        self.loads[lid] = h
        return h

    def load_ready(self, h) -> bool:
        """Rank 0: this rank's read is over (or never started)."""

        return h.done or h.err is not None or (h.lj is not None and h.lj.done.is_set())

    def complete_load(self, h) -> bool:
        """Rank 0: LOADED to rank 1, then both agree (``_finish_load``). True when the stored prompt is kept now."""

        if h.done:
            return getattr(h, "ok", False)
        self._check()
        self._emit(LOADED, [h.lid])
        self._flush()
        try:
            return self._finish_load(h.lid)
        except Exception as exc:              # a failed all-gather: the ranks may no longer agree
            self.broken = exc
            raise

    def _finish_load(self, lid: int) -> bool:
        """Both ranks: wait for this rank's read, agree through one all-gather that every read worked, then the
        extent holds a kept prompt; else the extent goes and the entry is forgotten on both."""

        h = self.loads.pop(lid)
        ok, snap, meta = 0, None, None
        if h.err is None:
            try:
                snap, _, meta = self.disk.finish_load(h.lj)
                ok = int(snap is not None and int(meta["n"]) == h.n)
            except Exception as exc:             # noqa: BLE001  (a failed read: every rank prefills)
                print(f"[tensorfold] spill rank {self.rank}: restore failed: {exc!r}", flush=True)
        else:
            print(f"[tensorfold] spill rank {self.rank}: restore failed: {h.err!r}", flush=True)
        every = self.g._gather_ints([ok]) if self.w.world > 1 else [[ok]]
        x = h.x
        x.loading = None
        h.done = True
        h.ok = all(r[0] for r in every)
        if not h.ok:
            self.disk.stats.load_failed += 1
            self.disk.forget(h.key)
            self.pool.remove(x)
            return False
        ids = [int(t) for t in json.loads(meta["tokens"])]
        keyed = json.loads(meta["keyed"]) if meta.get("keyed") else None
        snap.ids = ids
        snap.pending, snap.mtp_len, snap.rows, snap.nbytes = None, -1, None, 0
        snap.kid = self.next_kid
        self.next_kid += 1
        snap.key = keyed if keyed is not None else ids if self.rank == 0 else None
        snap.images = keyed is not None
        snap.shared = False
        snap.own, snap.chat = False, 0           # a restored state counts against no conversation's quota
        snap.extent = x
        x.kept.append(snap)
        self.kept.append(snap)
        while len(self.kept) > max(1, self.g.cache_entries):
            self._drop(cap_victim(self.kept))
        return True

    def flush_spill(self, budget: float | None = None) -> None:
        """Rank 0 (the scheduler's worker), a clean shutdown: the kept prompts not yet on disk, newest first, as many
        as the flush budget's time allows at the writers' rate so far (FLUSH carries the list), then both ranks wait
        for their writes."""

        if self.disk is None:
            return
        d = self.disk
        budget = d.cfg.flush_s if budget is None else budget
        s = d.stats
        rate = s.written_bytes / s.write_s if s.write_s > 1 and s.written_bytes else 0.5 * (1 << 30)
        rate *= d.cfg.writers
        kids, t = [], 0.0
        for c in reversed(self.kept):
            if not self._spillable(c) or d.has(self._ident(c), self._key(c)) or self._superseded(c):
                continue
            nb = sum(t2.numel() * t2.element_size() for _, t2 in self._planes(c.extent.base, len(c.ids)))
            t += nb / rate
            if t > 0.8 * budget:
                break
            kids.append(c.kid)
        self._emit(FLUSH, kids)
        if not self.lanes:                    # rank 1 goes back to the doorbell after the FLUSH
            self._emit(IDLE, [])
        self._flush()
        if not self.lanes:
            self.idle = True
        self._do_flush(kids, budget)

    def _do_flush(self, kids, budget: float | None = None) -> None:
        t = time.perf_counter()
        budget = self.disk.cfg.flush_s if budget is None else budget
        done: list = []
        for kid in kids:
            c = next((k for k in self.kept if k.kid == kid), None)
            if c is not None:                    # newest first: the flush's own earlier (newer) picks stay
                self._spill(c, keep=frozenset(self._key(k) for k in done))
                done.append(c)
        ok = self.disk.drain(max(1.0, budget - (time.perf_counter() - t)))
        print(f"[tensorfold] spill rank {self.rank}: flushed {len(done)} kept prompts in "
              f"{time.perf_counter() - t:.1f}s{'' if ok else ' (not finished)'}", flush=True)

    # -- room --------------------------------------------------------------------------------------------------------------
    def _used(self, x) -> int:
        used = max([len(k.ids) for k in x.kept], default=0)
        if getattr(x, "loading", None) is not None:      # a stored prompt being read into it (``_start_load``)
            used = max(used, x.loading.n)
        if x.owner is not None:
            used = max(used, self.lanes[x.owner].st.pos)
        if self.partial is not None:
            used = max([used] + [stop for lane, stop in self.partial.group if lane.extent is x])
        return used

    def _move(self, x, base: int, size: int) -> None:
        """Both ranks: relocate extent x (its used rows copied) to [base, base + size)."""

        used = self._used(x)
        old = self.pool.move(x, base, size)
        self.arena.copy(old, base, used)
        if x.owner is not None:
            self.lanes[x.owner].st.bind(x.base, x.size)

    def _resize(self, x, size: int) -> None:
        self.pool.resize(x, size)
        if x.owner is not None:
            self.lanes[x.owner].st.bind(x.base, x.size)

    def _victims(self, protect) -> list:
        keep = set(id(x) for x in protect)
        return [c for c in self.kept if c.extent.owner is None and id(c.extent) not in keep]

    def _compact(self, grow=None) -> None:
        """Rank 0: slide every extent down to the lowest free rows (MOVEs), so the free rows are one range; with
        ``grow``, only the extents up to it slide down and the ones after it up to the pool's end, so that range
        follows ``grow`` (each extent moves once at most)."""

        at = 0
        for x in list(self.pool.extents):
            if grow is not None and x.base > grow.base:
                break
            if x.base != at:
                self._emit(MOVE, [x.eid, at, x.size])
                self._move(x, at, x.size)
            at += x.size
        if grow is None:
            return
        top = self.pool.rows
        for x in reversed(list(self.pool.extents)):
            if x is grow:
                break
            top -= x.size
            if x.base != top:
                self._emit(MOVE, [x.eid, top, x.size])
                self._move(x, top, x.size)

    def _room(self, need: int, protect=()) -> int | None:
        """Rank 0: a base with ``need`` free rows, evicting kept prompts (least recently used first, never
        ``protect``'s extents) only while the free rows fall short of ``need``, then compacting if they are not
        together: moving rows costs far less than prefilling an evicted conversation again. None if even evicting
        every other kept prompt leaves too little."""

        base = self.pool.place(need)
        while base is None:
            if self.pool.free_rows() >= need:
                self._compact()
                return self.pool.place(need)
            victims = self._victims(protect)
            if not victims:
                break
            self._evict(victims[0])
            base = self.pool.place(need)
        return base

    def _grow(self, x, size: int, protect=()) -> bool:
        """Rank 0: extent x to at least ``size`` rows, in place or moved; as in ``_room``, compacting before evicting
        more than the free rows fall short."""

        size = align_up(size)
        if size <= x.size:
            return True
        protect = list(protect) + [x]
        while True:
            if self.pool.room_after(x) >= size - x.size:
                self._emit(GROW, [x.eid, size])
                self._resize(x, size)
                return True
            base = self.pool.place(size, ignore=[x])
            if base is not None:
                self._emit(MOVE, [x.eid, base, size])
                self._move(x, base, size)
                return True
            if self.pool.free_rows() + x.size >= size:
                self._compact(grow=x)            # the rows are free, just not together: gathered right after x
                continue
            victims = self._victims(protect)
            if not victims:
                return False
            self._evict(victims[0])

    # -- iterations ---------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def round(self) -> list[Stream]:
        """Rank 0: one prompt chunk or one decode round; returns the streams that finished."""

        self._check()
        _watch(self.watchdog, "iteration")               # cancelled at the end of ``finish``
        self.iteration_since = time.monotonic()           # /health's iteration_s
        try:
            if self.disk is not None and self.disk.cfg.highwater < 1.0:
                t0 = time.perf_counter()
                self._behind()
                self.disk.stats.behind_s += time.perf_counter() - t0
            return self._iterate()
        except Exception as exc:
            self.broken = exc
            raise

    def _fill_turn(self) -> Lane | None:
        filling = [l for l in self.lanes.values() if not l.decoding]
        if not filling:
            return None
        decoding = [l for l in self.lanes.values() if l.decoding and not l.paused and not l.s.done]
        if decoding and self.fill_budget and self.decode_due:
            self.decode_due = False
            return None
        if self.partial is not None:
            return self.partial.group[0][0]
        if decoding and not self.fill_budget:
            self.credit += self.share
            if self.credit < 1:
                return None
            self.credit -= 1
        s = next_fill([l.s for l in filling])
        return self.lanes[s.sid]

    def _iterate(self) -> list[Stream]:
        # a filling stream whose client has left ends before its next prompt chunk (sliced or not; rank 0 polls, the
        # FINISH it sends makes every rank drop it): a dropped long prompt frees the server within a chunk
        cancelled = []
        for lane in self.lanes.values():
            poll = getattr(lane.s.emit, "cancelled", None)
            if not lane.decoding and not lane.s.done and poll is not None and poll():
                lane.s.done = True
                lane.s.started = lane.s.finished = time.perf_counter()
                cancelled.append(lane.s)
        if cancelled:
            return cancelled                     # finish() mirrors cancellation and retires partial groups
        prof = self.profiles[False] if self.profiles is not None else None      # prompt chunks count there
        lane = self._fill_turn()
        if self.partial is not None and lane is not None:
            return self._part_iteration()
        group = self._group(lane) if lane is not None and self.group else None
        busy = any(l.decoding and not l.s.done and not l.paused for l in self.lanes.values())
        if lane is not None and busy and self.fill_budget and lane.head is None:
            self.partial = sliced_fill.PendingFill(group if group is not None else [(lane, self._stop(lane))],
                                                  many=group is not None)
            return self._part_iteration()
        if group is not None:
            t = time.perf_counter()
            rows = sum(stop - l.st.pos for l, stop in group)
            self._emit(MFILL, [len(group), *[v for l, stop in group for v in (l.sid, stop)]])
            self._flush()
            firsts = self._fill_many(group)
            for (l, _), first in zip(group, firsts):
                if first is not None:
                    l.s.take([first], self._ends(l))
            self.grouped["chunks"] += 1
            self.grouped["pieces"] += len(group)
            self.grouped["rows"] += rows
            if prof is not None:
                prof.fill(time.perf_counter() - t)
            return [l.s for l, _ in group if l.s.done]
        if lane is not None:
            t = time.perf_counter()
            stop = self._stop(lane)
            self._emit(FILL, [lane.sid, stop])
            self._flush()
            first = self._fill(lane, stop)
            if first is not None:
                lane.s.take([first], self._ends(lane))
            if prof is not None:
                prof.fill(time.perf_counter() - t)
            return [lane.s] if lane.s.done else []
        lanes = [l for l in self.lanes.values() if l.decoding and not l.s.done]
        t_begin = time.perf_counter()
        for l in lanes:                       # room for this round's widest window, else paused
            l.paused = not self._grow(l.extent, l.st.pos + self.rows_max, protect=[l.extent])
        active = [l for l in lanes if not l.paused]
        if not active:
            done = []
            if lanes and not any(not l.decoding for l in self.lanes.values()):
                done = self._give_back()      # every stream waits for room: the youngest goes back to the queue
            if self.outbox:                   # what was decided (evictions, moves, a stream given back) goes now
                self._flush()
            return done
        lone = False
        if (self.tune.lone and len(active) == 1 and len(self.lanes) == 1 and self.lone_rows
                and active[0].s.count - len(active[0].s.out) >= LONE_LEFT):
            lone = self._home(active[0])
        prof = self.profile = self.profiles[lone] if self.profiles is not None else None
        if prof is not None:
            prof.begin(t_begin)
        plan = [(l.sid, l.depth) for l in active]
        short = self.short_round_due
        self._emit(ROUND, [len(plan), *[v for p in plan for v in p], int(lone), int(short)])
        if prof is not None:
            prof.mark("plan")
        self._flush()
        self.short_round_due = False
        if prof is not None:
            prof.mark("message")
        self._round(active, lone=lone, short=short)
        return [l.s for l in active if l.s.done]

    def _part_iteration(self) -> list[Stream]:
        pending = self.partial
        lo = pending.layer
        busy = any(l.decoding and not l.paused and not l.s.done for l in self.lanes.values())
        hi = self.fill_costs.stop(pending.rows, lo, self.fill_budget) if busy else len(self.w.layers)
        if pending.many:
            payload = [len(pending.group), lo, hi, *[v for lane, stop in pending.group for v in (lane.sid, stop)]]
            self._emit(MFILLPART, payload)
        else:
            lane, stop = pending.group[0]
            self._emit(FILLPART, [lane.sid, stop, lo, hi])
        self._flush()
        done = self._part(pending.group, lo, hi, pending.many)
        pending.layer_ms += pending.slice_layer_ms
        if hi == len(self.w.layers):
            tail = max(0.0, pending.slice_wall_ms - pending.slice_layer_ms - self.fill_costs.overhead)
            self.fill_costs.observe_tail(pending.rows, tail)
            self.fill_costs.observe(pending.rows, 0, hi, pending.layer_ms)
        else:
            self.fill_costs.observe_overhead(max(0.0, pending.slice_wall_ms - pending.slice_layer_ms))
        self.decode_due = True
        self.short_round_due = not self.fill_drafts
        return done

    @torch.no_grad()
    def _part(self, group, lo: int, hi: int, many: bool) -> list[Stream]:
        """Both ranks: apply one contiguous slice, including publication only on its final layer."""
        if not group or any(l.decoding or stop != self._stop_bound(l, stop) for l, stop in group):
            raise RuntimeError("a prompt slice contains a decoding stream or crosses a kept point")
        if self.partial is None:
            if lo:
                raise RuntimeError("a resumed prompt slice has no pending chunk")
            self.partial = sliced_fill.PendingFill(list(group), many=many)
        pending = self.partial
        if pending.many != many or [(l.sid, stop) for l, stop in group] != [(l.sid, stop) for l, stop in pending.group]:
            raise RuntimeError("a prompt slice changed its chunk")
        t = time.perf_counter()
        begin = torch.cuda.Event(enable_timing=True)
        pending.layer_event = torch.cuda.Event(enable_timing=True)
        begin.record()
        before = [lane.s.prefill_s for lane, _ in group]
        if many:
            firsts = self._fill_many(group, layers=(lo, hi))
        else:
            lane, stop = group[0]
            firsts = [self._fill(lane, stop, layers=(lo, hi))]
        if hi < len(self.w.layers):
            pending.layer_event.record()
        # finish() joins HC overlap; drain all device work before the next iteration can run decode graphs.
        torch.cuda.synchronize(self.w.device)
        dt = time.perf_counter() - t
        pending.slice_wall_ms = dt * 1000
        pending.slice_layer_ms = begin.elapsed_time(pending.layer_event)
        for (lane, _), elapsed in zip(group, before):
            lane.s.prefill_s = elapsed + dt           # include the slice's GPU execution, not just its enqueue
        for (lane, _), first in zip(group, firsts):
            if first is not None and lane.sid not in pending.retired:
                lane.s.take([first], self._ends(lane))
        done = [l.s for l, _ in group if l.s.done and l.sid not in pending.retired]
        if hi == len(self.w.layers):
            pending.close()
            self.partial = None
            for sid in pending.retired:
                self._finish(sid)
            if many and self.rank == 0:
                self.grouped["chunks"] += 1
                self.grouped["pieces"] += len(group)
                self.grouped["rows"] += pending.rows
        return done

    def _home(self, lane: Lane) -> bool:
        """Rank 0: a lone stream into the one-stream graphs' home, slot 0 at the pool's first rows (SLOT, then
        kept prompts in the way evicted and its extent MOVEd); True once it is there."""

        if self.e.graphed(lane.st):
            return True
        if lane.slot != 0:
            self._emit(SLOT, [lane.sid, 0])
            self._reslot(lane, 0)
        x = lane.extent
        if x.base != 0:
            for c in list(self.kept):                  # kept prompts in [0, x.size) make way
                y = c.extent
                if y is not x and y.base < x.size and c in self.kept:
                    self._evict(c)
            if any(y is not x and y.base < x.size for y in self.pool.extents):
                return False                           # something else holds the home rows (cannot happen alone)
            self._emit(MOVE, [x.eid, 0, x.size])
            self._move(x, 0, x.size)
        return self.e.graphed(lane.st)

    def _reslot(self, lane: Lane, slot: int) -> None:
        """Both ranks: the lane's slot state (KDA states, conv windows, DFlash2 context) copied into ``slot``."""

        from .forward import State

        e, old = self.e, lane.slot
        if any(l.slot == slot for l in self.lanes.values() if l is not lane):
            raise RuntimeError(f"stream {lane.sid} cannot take slot {slot}: another stream holds it")
        e.slots.rec[slot].copy_(e.slots.rec[old])
        e.slots.conv[slot].copy_(e.slots.conv[old])
        if self.drafts is not None:
            a, b = self.drafts.contexts[old], self.drafts.contexts[slot]
            for dst, src in zip(list(b.kc) + list(b.vc), list(a.kc) + list(a.vc)):
                dst.copy_(src)
            b.context_end = a.context_end
            b.pos_dev.copy_(a.pos_dev)
        st = State(self.w, lane.extent.size, e.rows, caches=e.caches, base=lane.extent.base, slots=e.slots, slot=slot)
        st.set_pos(lane.st.pos)
        st.cur = list(lane.st.cur)
        lane.st, lane.slot = st, slot

    def _stop(self, lane: Lane) -> int:
        pos, n = lane.st.pos, len(lane.s.prompt)
        stop = next((p for p in lane.stops if p > pos), n)
        busy = any(l.decoding and not l.s.done for l in self.lanes.values())
        rows = self.fill_rows if busy or lane.s.background else self.e.prefill_rows
        return min(stop, pos + rows)

    def _group(self, first: Lane) -> list[tuple[Lane, int]] | None:
        """Rank 0, TF_GLM_MULTI_PREFILL=1: the (lane, stop) pieces of this iteration's prompt chunk when it can hold
        more than one stream's rows, else None (``first`` fills alone, as without the setting). The chunk's row budget
        is the single-stream one (TF_GLM_FILL_ROWS while others decode, else the engine's prompt chunk). Foreground
        streams before background ones, then the fewest prompt rows left first (a short prompt's first token does not
        wait behind a long prompt; the long one takes the rest of the budget), then admission order; each takes the
        rows to its next stop (a kept point or its end) when they fit, else the budget left in whole ``fill_unit``s.
        Image prompts and replays of a kept head fill alone."""

        def plain(l: Lane) -> bool:
            return not l.decoding and l.feed is None and l.head is None

        if not plain(first):
            return None
        busy = any(l.decoding and not l.s.done for l in self.lanes.values())
        left = self.fill_rows if busy else self.e.prefill_rows
        lanes = sorted((l for l in self.lanes.values() if plain(l)),
                       key=lambda l: (bool(l.s.background), len(l.s.prompt) - l.st.pos, l.order))
        out: list[tuple[Lane, int]] = []
        for l in lanes:
            pos, n = l.st.pos, len(l.s.prompt)
            stop = next((p for p in l.stops if p > pos), n)
            room = min(left, self.fill_rows if l.s.background else left)
            if stop - pos > room:
                stop = pos + room // self.unit * self.unit
            if stop <= pos:
                continue
            out.append((l, stop))
            left -= stop - pos
            if left <= 0:
                break
        return out if len(out) > 1 else None

    def _ends(self, lane: Lane) -> tuple[int, ...]:
        return self.eos if lane.s.stop_eos else ()

    @torch.no_grad()
    def _fill(self, lane: Lane, stop: int, *, layers=None) -> int | None:
        """Both ranks: the lane's prompt rows pos .. stop in one chunk; its kept states; at the prompt's end its first
        token (the stream then decodes)."""

        from .decode import prefill_chunk, prefill_chunk_steps

        e = self.e
        s = lane.s
        prompt = s.prompt
        n = len(prompt)
        start = lane.st.pos
        replay = lane.head is not None           # resumed at the prompt's end: no rows, the kept head's row
        if not (start < stop <= n or replay and start == stop == n) or stop - start > e.prefill_rows:
            raise RuntimeError(f"stream {lane.sid}: a prompt chunk {start} .. {stop} of {n}")
        t = time.perf_counter()
        prev = e.use(lane.st)
        try:
            if replay:
                out, lane.head = lane.head, None
            else:
                if layers is None:
                    out = prefill_chunk(e, prompt, start, stop - start, drafter=self._ctx(lane) if lane.dflash else None,
                                        feed=lane.feed, mtp=False, head=stop == n)
                else:
                    complete, out = sliced_fill.advance(self.partial, *layers, len(self.w.layers),
                        lambda: prefill_chunk_steps(e, prompt, start, stop - start,
                            drafter=self._ctx(lane) if lane.dflash else None, feed=lane.feed, head=stop == n),
                        before_finalize=self.partial.layer_event.record)
                    if not complete:
                        return None
            if stop < n:
                if stop in lane.stops:
                    self._keep(lane, stop, own=stop == lane.point)
                return None
            if lane.point == n and not replay:
                self._keep(lane, n, head=out[:1].clone(), own=True)    # before a grammar's mask writes into the row
            logits = out[:1]
            if lane.constraint is not None:             # the first token's row, under the reply's grammar
                lane.constraint.mask(logits, lane.constraint.window([0], [-1]), self.w.vocab_offset)
            first = sample_streams(self.w, [(logits, [n], s.sampling)])[0][0]
            if lane.constraint is not None:
                lane.constraint.advance([first])
        finally:
            e.use(prev)
            s.prefill_s += time.perf_counter() - t
        self._started(lane, first)
        return first

    def _started(self, lane: Lane, first: int) -> None:
        """Both ranks: the lane's prompt is in and its first token sampled: it decodes from the next round."""

        from .copy_drafts import CopyDrafts

        s = lane.s
        prompt = s.prompt
        lane.decoding = True
        s.started = time.perf_counter()
        if lane.policy is not None and self.g.copy is not None:
            c = self.g.copy                 # no pads: batched windows are not the single-stream widths
            lane.copies = CopyDrafts(list(prompt) + [first], c.match, c.most, prompt=len(prompt),
                                     reply_match=c.reply_match, miss_most=c.miss_most)
        if lane.policy is not None:
            lane.depth = min(lane.policy.next(0, 0), s.count - 1)

    @torch.no_grad()
    def _fill_many(self, group: Sequence[tuple[Lane, int]], *, layers=None) -> list[int | None]:
        """Both ranks, TF_GLM_MULTI_PREFILL: every (lane, stop)'s prompt rows pos .. stop in ONE chunk
        (``multi_prefill.prefill_pieces``: each lane's rows the bits of a chunk of their own); then per lane, in
        order, what ``_fill`` does after its chunk: its kept states, and at its prompt's end its first token (all
        first tokens sampled together, each as alone: ``sample_streams``). Returns each lane's first token or None."""

        e = self.e
        pieces = []
        for lane, stop in group:
            start, n = lane.st.pos, len(lane.s.prompt)
            if lane.decoding or lane.feed is not None or lane.head is not None:
                raise RuntimeError(f"stream {lane.sid} cannot join a multi-prompt chunk")
            if not start < stop <= n or stop - start > e.prefill_rows:
                raise RuntimeError(f"stream {lane.sid}: a prompt chunk {start} .. {stop} of {n}")
            pieces.append(multi_prefill.Piece(lane.st, list(lane.s.prompt[start:stop]),
                                              drafter=self._ctx(lane) if lane.dflash else None, head=stop == n))
        t = time.perf_counter()
        if layers is None:
            heads = multi_prefill.prefill_pieces(e, pieces)
        else:
            complete, heads = sliced_fill.advance(self.partial, *layers, len(self.w.layers),
                                                   lambda: multi_prefill.prefill_piece_steps(e, pieces),
                                                   before_finalize=self.partial.layer_event.record)
            if not complete:
                dt = time.perf_counter() - t
                for lane, _ in group:
                    lane.s.prefill_s += dt
                return [None] * len(group)
        parts, who = [], []
        for k, ((lane, stop), out) in enumerate(zip(group, heads)):
            if self.partial is not None and lane.sid in self.partial.retired:
                continue
            n = len(lane.s.prompt)
            if stop < n:
                if stop in lane.stops:
                    self._keep(lane, stop, own=stop == lane.point)
                continue
            if lane.point == n:
                self._keep(lane, n, head=out[:1].clone(), own=True)    # before a grammar's mask writes into the row
            logits = out[:1]
            if lane.constraint is not None:             # the first token's row, under the reply's grammar
                lane.constraint.mask(logits, lane.constraint.window([0], [-1]), self.w.vocab_offset)
            parts.append((logits, [n], lane.s.sampling))
            who.append(k)
        firsts: list[int | None] = [None] * len(group)
        if parts:
            for k, rows in zip(who, sample_streams(self.w, parts)):
                lane = group[k][0]
                firsts[k] = rows[0]
                if lane.constraint is not None:
                    lane.constraint.advance([rows[0]])
        dt = time.perf_counter() - t
        for (lane, _), first in zip(group, firsts):
            lane.s.prefill_s += dt
            if first is not None:
                self._started(lane, first)
        return firsts

    @torch.no_grad()
    def _round(self, lanes: list[Lane], depths: list[int] | None = None, lone: bool = False,
               short: bool = False) -> None:
        """Both ranks: one decode round over ``lanes`` (``depths``: rank 0's, checked on rank 1); ``lone``: one
        stream at home, verified through the one-stream graphs when its window fits them. ``short``: between fill
        slices, commit one token per stream without proposing drafts (the same keyed sample, lower round cost)."""

        from .decode import copy_room
        from .dflash2_multi import DraftRequest

        if depths is not None and [l.depth for l in lanes] != depths:
            raise RuntimeError(f"rank {self.rank}'s draft depths {[l.depth for l in lanes]} are not rank 0's {depths}")
        prof = self.profile
        tune = self.tune
        drafts: list[list[int]] = []
        copied: list[bool] = []
        asks = []
        for l in lanes:
            c = l.copies.propose(copy_room(l.copies, l.s.count, l.s.out)) if l.copies is not None and not short else []
            copied.append(bool(c))
            drafts.append(list(c))
            if not short and not c and l.dflash and l.depth > 0:
                conf = l.policy.confidence
                if tune.depth == "scale":
                    conf = scaled_confidence(conf, len(lanes), tune.alpha)
                asks.append((len(drafts) - 1, DraftRequest(self._ctx(l), l.s.out[-1], l.depth, l.s.sampling,
                                                           conf, **l.policy.chain_rule)))
        if asks and tune.depth == "joint":
            self._joint(lanes, drafts, asks)
        elif asks:
            got = self.drafts.propose([a for _, a in asks])
            for (k, _), d in zip(asks, got):
                drafts[k] = list(d)
        drafts = trim(drafts, self.window)
        if prof is not None:
            prof.mark("propose")
        segments = []
        for l, d in zip(lanes, drafts):
            tokens = [l.s.out[-1]] + d
            l.window = None
            if l.constraint is not None:                # the chain cut at its first draft the grammar rejects
                l.window = l.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                tokens = list(l.window.tokens)
            segments.append(Segment(l.st, tokens))
        verify = self.verify
        if lone:
            if not self.e.graphed(lanes[0].st):
                raise RuntimeError(f"rank {self.rank}: a lone round's stream is not at the graphs' home")
            if len(segments[0].tokens) <= self.lone_rows:
                verify = self.solo_verify
        if prof is not None:
            prof.gpu_start()
        v = forward_streams(verify, segments)
        if prof is not None:
            prof.gpu_stop()
            prof.mark("verify")
        parts = []
        for l, seg, logits in zip(lanes, segments, v.logits):
            if l.window is not None:
                l.constraint.mask(logits, l.window, self.w.vocab_offset)
            parts.append((logits, [l.st.pos + 1 + r for r in range(len(seg.tokens))], l.s.sampling))
        # A short round has one row per stream: pack nucleus gathers too, avoiding
        # repeated rank exchanges between prompt slices with the same exact draws.
        sampled = sample_packed(self.w, parts) if short or tune.sampler == "packed" else sample_streams(self.w, parts)
        if prof is not None:
            prof.mark("sample")
        keeps = []
        for l, seg, rows in zip(lanes, segments, sampled):
            keep = 1
            for i, d in enumerate(seg.tokens[1:]):
                if rows[i] != d or (l.s.stop_eos and rows[i] in self.eos):
                    break
                keep += 1
            keeps.append(keep)
        if prof is not None:
            prof.mark("accept")
        verify.commit(segments, keeps)
        if prof is not None:
            prof.mark("commit")
        if self.drafts is not None:
            items = [(self._ctx(l), v.taps[k][:keep]) for k, (l, keep) in enumerate(zip(lanes, keeps)) if l.dflash]
            if items:
                self.drafts.commit(items)
        if prof is not None:
            prof.mark("taps")
        for l, seg, rows, keep, was_copy in zip(lanes, segments, sampled, keeps, copied):
            s = l.s
            new = rows[:keep]
            if l.constraint is not None:
                l.constraint.advance(new)
            if l.copies is not None:
                l.copies.extend(new)
            ndrafts = len(seg.tokens) - 1
            s.counted(len(seg.tokens))                     # Stream.take counts the accepted drafts
            if was_copy:
                l.copy_rounds += 1
                l.copy_drafted += ndrafts
                l.copy_accepted += keep - 1
            room = s.count - len(s.out)
            s.take(new[:max(0, room)], self._ends(l))
            if short:
                l.depth = min(l.depth, max(0, s.count - len(s.out)))
            elif l.policy is not None:
                l.depth = min(l.policy.next(*((0, 0) if was_copy else (ndrafts, keep - 1))),
                              max(0, s.count - len(s.out)))
        if prof is not None:
            prof.mark("emit")
            prof.end(len(lanes), sum(len(g.tokens) for g in segments), sum(keeps),
                     sum(len(g.tokens) - 1 for g in segments))

    def _joint(self, lanes: list[Lane], drafts: list[list[int]], asks: list) -> None:
        """TF_GLM_MULTI_DEPTH=joint: every asking stream's chain walked to its depth in one block pass, then the
        round's rows allocated across them by expected tokens per ms (``multi_tune.allocate``). The picks are each
        policy's; only where each chain stops differs (drafts never change a reply)."""

        md = self.drafts
        live = [(k, r) for k, r in asks if min(r.depth, md.block - 1) >= 1 and r.ctx.context_end > 0]
        if not live:
            return
        got = md.candidates([(r.ctx, r.pending, min(r.depth, md.block - 1)) for _, r in live])
        reach = []
        for (k, r), (tokens, values, proj) in zip(live, got):
            confs: list[float] = []
            drafts[k] = md.d.chain(tokens, values, proj, r.pending, r.ctx.context_end + 1, r.sampling, 0.0,
                                   beta=r.beta, confs=confs)
            reach.append(reach_of(confs))
        asked = {k for k, _ in live}
        fixed = sum(1 + (0 if k in asked else len(d)) for k, d in enumerate(drafts))
        take = allocate(reach, fixed, self._row_cost, self.tune.overhead_ms, self.window)
        for (k, _), n in zip(live, take):
            drafts[k] = drafts[k][:n]

    # -- ends ----------------------------------------------------------------------------------------------------------------
    def finish(self, done: list[Stream]) -> None:
        """Rank 0: finished (or cancelled) streams leave on both ranks; an idle rank 1 is told to wait for the bell."""

        for s in done:
            lane = self.lanes.get(s.sid)
            if lane is None or lane.s is not s:
                continue
            reason = CANCELLED if s.error is None and len(s.out) < s.count and not (
                s.out and s.out[-1] in self._ends(lane)) else DONE
            self._emit(FINISH, [lane.sid, reason])
            self._finish(lane.sid)
        if not self.lanes and not self.idle and self.broken is None:
            self.decode_due = self.short_round_due = False
            self._emit(IDLE, [])
            self._flush()
            self.idle = True
        _unwatch(self.watchdog)
        self.iteration_since = None

    def _finish(self, sid: int) -> None:
        """Both ranks: the stream's slot frees; its extent stays for its kept prompts (shrunk to them) or goes."""

        pending = self.partial
        if pending is not None and pending.contains(sid):
            pending.retired.add(sid)
            if len(pending.retired) < len(pending.group):
                self.lanes[sid].s.done = True
                return                              # reserved slot/extent until the shared forward completes
            pending.close()
            torch.cuda.synchronize(self.w.device)
            self.partial = None
            for retired in pending.retired - {sid}:
                self._finish(retired)
        lane = self.lanes.pop(sid)
        x = lane.extent
        x.owner = None
        self._settle(x)
        lane.s.finished = lane.s.finished or time.perf_counter()

    def _give_back(self) -> list[Stream]:
        """Rank 0: the youngest stream (one without a grammar or images, which cannot replay) leaves its slot and
        extent and returns to the queue; it replays later, its sent tokens owed, not sent again. If every stream has
        a grammar or images, the youngest fails instead (returned, to be finished)."""

        lanes = sorted(self.lanes.values(), key=lambda l: l.order, reverse=True)
        lane = next((l for l in lanes if l.constraint is None and not l.images), None)
        if lane is None:
            lane = lanes[0]
            lane.s.error = NoRoom("the pool is full: every stream waits for room, and the youngest cannot replay "
                                  "(a grammar or images)")
            lane.s.done = True
            return [lane.s]
        for c in list(lane.extent.kept):
            self._evict(c)
        self._emit(FINISH, [lane.sid, REQUEUED])
        self._finish(lane.sid)
        self.requeue.append(lane.s)
        return []

    def drop(self) -> list[Stream]:
        """After an error in an iteration: forget every stream (the ranks can no longer be trusted to agree)."""

        live = [l.s for l in self.lanes.values()]
        if self.partial is not None:
            self.partial.close()
            self.partial = None
        self.lanes = {}
        if self.broken is None:
            self.broken = RuntimeError("an iteration failed")
        return live

    def stats(self, s: Stream) -> dict:
        """A finished request's stats (as ``GlmEngine.generate`` reports them)."""

        out = s.stats()
        out["decode_s"] = round(max((s.finished or time.perf_counter()) - s.started, 0.0), 4)
        tokens = list(s.out[:s.count])
        out.update(sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
                   tokens_per_round=round((len(tokens) - 1) / max(s.rounds, 1), 3), parallel=self.count)
        lane_info = getattr(s, "glm_lane", None)
        if lane_info:
            out.update(lane_info)
        return out

    # -- rank 1 --------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def follow(self, once: bool = False) -> None:
        """Rank 1: apply rank 0's messages forever (``once``: one message)."""

        while True:
            if self.idle:
                self.g._await_bell()
                self.idle = False
            _watch(self.watchdog, f"rank {self.rank} message")   # busy: rank 0's next message is due within a round
            self.apply(unseal(self.g._share(None), self.received, self.rank))
            self.received = (self.received + 1) % SEAL_MOD
            _unwatch(self.watchdog)
            if once:
                return

    def apply(self, msg: Sequence[int]) -> None:
        """Rank 1: one iteration's ops, in order."""

        from tensorfold.engine.exact_sampling import Sampling  # noqa: F401

        for op, p in self.parse(msg):
            if op == ADMIT:
                a = self._parse_admit(p)
                s = Stream(list(a["prompt"]), a["count"], a["sampling"], draft=a["draft"], stop_eos=a["stop_eos"],
                           sid=a["sid"])
                constraint = None
                if a["packed"]:
                    from tensorfold.engine import grammar

                    constraint = grammar.compiler(self.g, self.g.model_dir, self.g.eos).follow(a["packed"])
                    s.constraint = constraint
                feed = None
                if a["positions"]:
                    from .engine import VisionFeed

                    feed = VisionFeed(self.g, list(a["positions"]), None)
                    s.vision = feed
                self._admitted(a, s, feed, constraint)
            elif op == EVICT:
                self._drop(self._kept_by_id(p[0]), spill=bool(p[1]) if len(p) > 1 else True)
            elif op == LOAD:
                self._start_load(*p)
            elif op == LOADED:
                self._finish_load(*p)
            elif op == TOUCH:
                self.disk.touch_key((p[0], p[1]))
            elif op == FLUSH:
                self._do_flush(p)
            elif op == SPILL:
                self._spill(self._kept_by_id(p[0]), keep=self._kept_keys())
            elif op == MOVE:
                self._move(self.pool.get(p[0]), p[1], p[2])
            elif op == GROW:
                self._resize(self.pool.get(p[0]), p[1])
            elif op == FILL:
                lane = self.lanes[p[0]]
                if p[1] != self._stop_bound(lane, p[1]):
                    raise RuntimeError(f"rank 1: stream {p[0]}'s prompt chunk to {p[1]} passes a kept point")
                first = self._fill(lane, p[1])
                if first is not None:
                    lane.s.take([first], self._ends(lane))
            elif op == MFILL:
                group = []
                for sid, stop in zip(p[1:1 + 2 * p[0]:2], p[2:2 + 2 * p[0]:2]):
                    lane = self.lanes[sid]
                    if stop != self._stop_bound(lane, stop):
                        raise RuntimeError(f"rank 1: stream {sid}'s prompt chunk to {stop} passes a kept point")
                    group.append((lane, stop))
                for (lane, _), first in zip(group, self._fill_many(group)):
                    if first is not None:
                        lane.s.take([first], self._ends(lane))
            elif op == FILLPART:
                sid, stop, lo, hi = p
                self._part([(self.lanes[sid], stop)], lo, hi, False)
            elif op == MFILLPART:
                n, lo, hi = p[:3]
                group = [(self.lanes[sid], stop) for sid, stop in zip(p[3::2], p[4::2])]
                self._part(group, lo, hi, True)
            elif op == ROUND:
                n = p[0]
                sids, depths = p[1:1 + 2 * n:2], p[2:2 + 2 * n:2]
                lone = bool(p[1 + 2 * n]) if len(p) > 1 + 2 * n else False
                short = bool(p[2 + 2 * n]) if len(p) > 2 + 2 * n else False
                self._round([self.lanes[sid] for sid in sids], depths, lone=lone, short=short)
            elif op == SLOT:
                self._reslot(self.lanes[p[0]], p[1])
            elif op == FINISH:
                self._finish(p[0])
            elif op == IDLE:
                if self.lanes:
                    raise RuntimeError(f"rank 1 holds {len(self.lanes)} streams rank 0 has finished")
                self.idle = True
            else:
                raise RuntimeError(f"rank 1: unknown op {op}")

    def _stop_bound(self, lane: Lane, stop: int) -> int:
        """The stop rank 0 sent, if it passes no kept point (else where it should have stopped)."""

        nxt = next((q for q in lane.stops if q > lane.st.pos), len(lane.s.prompt))
        return min(stop, nxt)


# -- the scheduler ----------------------------------------------------------------------------------------------------
class GlmScheduler(Scheduler):
    """``cuda.scheduler.Scheduler`` for the GLM decoder: a request's GLM settings ride on its stream (``glm``); a
    request the pool cannot place yet waits at the head of the queue (``MultiDecoder.fits``); streams the decoder gives
    back (every stream waiting for room) return to the queue to replay."""

    class _Flush:                        # a shutdown flush's wake-up item in the queue (``flush_spill``), the class
        background = False               # itself; ``emit`` None: code that scans the queue for streams passes it by
        emit = None

    def __init__(self, decoder: MultiDecoder, *, max_streams: int) -> None:
        self.held = None
        self.flushing = None             # a shutdown's flush (``flush_spill``), taken first thing next iteration
        self.flush_lock = threading.Lock()
        self.loading: list = []          # (read, item): requests waiting for a stored prompt read (--spill-gib)
        self.ready: list = []            # items whose read ended: admitted before the queue
        self.staged: set = set()         # streams a read was looked for already
        # TF_GLM_MULTI_PREFILL_WAIT_MS: an idle server's wait for requests arriving with the first (their prompts then
        # share its first chunk); only with TF_GLM_MULTI_PREFILL=1
        self.gather_s = multi_prefill.wait_ms() / 1000.0 if getattr(decoder, "group", False) else 0.0
        # TFCAP-0075: the cap is operator configuration (TF_GLM_MAX_QUEUED, exported by scripts/config.sh):
        # unset queues as every scheduler always has; N refuses past the lanes plus N. vLLM's default is
        # likewise no limit — its own words, the cap "bounds vLLM's otherwise unbounded request queue".
        # A single-instance LAN deployment has no load balancer to shed load ahead of the server, so an
        # exposed one should set a small cap (ours: 0 — nothing waits past the lanes).
        queued = os.environ.get("TF_GLM_MAX_QUEUED")
        super().__init__(decoder, max_streams=max_streams,
                         max_in_system=max_streams + int(queued) if queued else None)

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, glm: dict | None = None) -> dict:
        self._check_admission(background)
        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background)
        stream.glm = dict(glm or {})
        # The caller polls; only the scheduler removes cancelled queued requests.
        stop, disconnected, cancel_error = [False], [False], [None]
        poll = getattr(emit, "cancelled", None)

        def cancelled() -> bool:
            if not disconnected[0] and cancel_error[0] is None and poll is not None:
                try:
                    if poll():                      # sticky: an older False poll must never undo another's True
                        disconnected[0] = stop[0] = True
                except Exception as exc:
                    cancel_error[0], stop[0] = exc, True
            return disconnected[0] or cancel_error[0] is not None

        stream.emit = lambda new: (box.put(("tokens", new)), stop[0])[1]
        stream.emit.cancelled = cancelled
        stream.emit.cancel_requested = lambda: disconnected[0] or cancel_error[0] is not None
        stream.emit.cancel_error = lambda: cancel_error[0]
        self.waiting.put((stream, box))
        while True:
            cancelled()
            try:
                kind, value = box.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "tokens":
                if not stop[0]:
                    try:
                        if emit(value):
                            stop[0] = True           # a normal stop/gate cut is not cancellation
                    except Exception as exc:
                        cancel_error[0], stop[0] = exc, True
            elif kind == "error":
                raise value
            else:
                if cancel_error[0] is not None:      # report callback failures only after active decoder cleanup
                    raise cancel_error[0]
                if disconnected[0]:
                    from tensorfold.server.cancellation import RequestCancelled
                    raise RequestCancelled("the client left during the reply")
                return value

    def _cancel_item(self, item) -> bool:
        """Scheduler-owned cancellation of a not-admitted request; no decoder/rank state exists yet."""
        from tensorfold.server.cancellation import RequestCancelled

        stream, box = item
        poll = getattr(stream.emit, "cancelled", None)
        if poll is None or not poll():
            return False
        error = getattr(stream.emit, "cancel_error", lambda: None)()
        box.put(("error", error if error is not None else RequestCancelled("the client left before admission")))
        self.staged.discard(id(stream))          # (--spill-gib: done looking for its stored prompt)
        return True

    def _check_admission(self, background: bool) -> None:
        """``Scheduler._check_admission`` (patch 0082), with requests whose stored prompt is being read or was read
        (--spill-gib) counted too: each holds the lane it will take."""

        if background or self.max_in_system is None:
            return
        held = self.decoder.live() + self._reading() + len(self.ready)
        if held + self.waiting.foreground_count() >= self.max_in_system:
            raise CapacityError("all serving lanes are busy; retry shortly")

    def _reading(self) -> int:
        """Requests whose stored prompt is being read (--spill-gib): each keeps the stream slot it will take."""

        return sum(1 for _, it in self.loading if it is not None)

    def _cancel_waiting(self) -> None:
        """Reap waiting/held requests even when every lane is busy; retain exact queue priorities/order."""
        import heapq

        self.ready = [it for it in self.ready if not self._cancel_item(it)]   # their stored prompts read (--spill-gib)
        self.loading = [(read, None if it is None or self._cancel_item(it) else it) for read, it in self.loading]
        if self.held is not None and self._cancel_item(self.held):
            self.held = None
        removed = []
        # Never call the socket callback while holding the queue lock: these are caller-latched flags.
        with self.waiting.mutex:
            kept = []
            for entry in self.waiting.queue:
                stream, _ = entry[2]
                flag = getattr(stream.emit, "cancel_requested", None)
                (removed if flag is not None and flag() else kept).append(entry)
            if removed:
                self.waiting.queue[:] = kept
                heapq.heapify(self.waiting.queue)
                self.waiting.not_full.notify_all()
        for entry in removed:
            self._cancel_item(entry[2])

    def _admit(self, first=None, until: float | None = None) -> list[Stream]:
        """Waiting requests into free slots while they fit; ``until`` (a ``time.monotonic`` deadline): an empty queue
        is waited on until then for more arrivals."""

        done: list[Stream] = []
        with self.flush_lock:
            flush, self.flushing = self.flushing, None
        if flush is not None:                        # a shutdown: kept prompts to the spill tier first
            try:
                self.decoder.flush_spill()
            finally:
                flush.set()
        for it in [it for it in self.loading if self.decoder.load_ready(it[0])]:
            self.loading.remove(it)
            try:
                self.decoder.complete_load(it[0])
            except Exception as exc:                 # noqa: BLE001  (the ranks may disagree now: this request fails)
                if it[1] is not None:
                    self.staged.discard(id(it[1][0]))
                    it[1][1].put(("error", exc))
                continue
            if it[1] is not None:                    # (None: cancelled during the read, answered then)
                self.ready.append(it[1])
        while self.decoder.live() + self._reading() < self.max_streams:
            if first is not None:
                item, first = first, None
            elif self.held is not None:                 # (before a read one: ``held`` is set only while empty)
                item, self.held = self.held, None
            elif self.ready:
                item = self.ready.pop(0)
            else:
                try:
                    left = until - time.monotonic() if until is not None else 0.0
                    item = self.waiting.get(timeout=left) if left > 0 else self.waiting.get_nowait()
                except queue.Empty:
                    break
            if self._cancel_item(item):
                continue
            stream, box = item
            if stream is self._Flush:                # (it woke the worker; ``flushing`` did the flush)
                continue
            if not self.decoder.fits(stream):
                self.held = item                     # waits, first in line, for room
                break
            prestage = getattr(self.decoder, "prestage", None)
            if prestage is not None and id(stream) not in self.staged:
                self.staged.add(id(stream))
                try:
                    read = prestage(stream)          # a stored prompt to read first: other streams go on meanwhile
                except Exception as exc:             # noqa: BLE001
                    self.staged.discard(id(stream))
                    box.put(("error", exc))
                    continue
                if read is not None:
                    self.loading.append((read, item))
                    continue
            self.staged.discard(id(stream))
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom:
                self.boxes.pop(id(stream))
                self.held = item
                break
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def flush_spill(self, timeout: float = 60.0) -> bool:
        """Any thread (a SIGTERM): the worker writes the kept prompts to the spill tier at its next iteration, whatever
        its slots and queue hold (the queued item only wakes an idle worker)."""

        done = threading.Event()
        with self.flush_lock:
            self.flushing = done
        self.waiting.put((self._Flush, done))
        return done.wait(timeout)

    def _yield(self) -> None:
        """``Scheduler._yield``, the stream's GLM settings carried to its replay."""

        if self.decoder.live() + self._reading() < self.max_streams or not self.waiting.foreground():
            return
        live = list(self.decoder.streams.values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes.pop(id(stream))
        self.decoder.finish([stream])
        self.yields += 1
        again = stream.continued()
        again.glm = getattr(stream, "glm", {})
        self.waiting.put((again, box))

    def _requeue(self) -> None:
        for stream in self.decoder.requeue:
            box = self.boxes.pop(id(stream), None)
            if box is None:
                continue
            again = stream.continued()
            again.glm = getattr(stream, "glm", {})
            self.yields += 1
            self.waiting.put((again, box))
        self.decoder.requeue = []

    def _loop(self) -> None:
        import sys
        import traceback

        while True:
            try:
                self._iteration()
            except Exception as exc:                 # noqa: BLE001  (never a silent dead worker: log, fail, go on)
                print(f"[tensorfold] --parallel scheduler iteration failed: {exc!r}", file=sys.stderr, flush=True)
                traceback.print_exc()
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)

    def _iteration(self) -> None:
        self._cancel_waiting()
        self._yield()
        reading = bool(self.loading or self.ready)
        idle = not self.decoder.live() and self.held is None and not reading
        if reading and not self.decoder.live():
            time.sleep(0.002)                    # only reads in flight: poll them
        first = self.waiting.get() if idle else None                # idle: wait for a request
        until = time.monotonic() + self.gather_s if idle and self.gather_s > 0 else None
        done = self._admit(first, until)
        try:
            done += self.decoder.round()
        except Exception as exc:                 # noqa: BLE001  (the live requests fail)
            for s in self.decoder.drop():
                self._reply(s, "error", exc)
        self._requeue()
        try:
            self.decoder.finish(done)
        except Exception as exc:                 # noqa: BLE001
            for s in self.decoder.drop():
                self._reply(s, "error", exc)
        for s in done:
            self._reply(s, *(("error", s.error) if s.error is not None else ("done", self.decoder.stats(s))))
        if self.held is not None and not self.decoder.live():
            time.sleep(0)                        # the held request retries next iteration (now alone)
        _unwatch(getattr(self.decoder, "watchdog", 0))
