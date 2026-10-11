"""A disk tier for kept prompt states on CUDA engines (``tensorfold serve --spill-gib``): a state that leaves memory
is written to local disk and read back when a later request extends the same tokens, also after a restart.

A family hands over the snapshot object it keeps (``encode``: tensors into the file, plain values into JSON, the class
by import path, as ``engine.prefix_snapshots`` does on MLX) and, for a shared pool, named row tensors read back into
caller-placed views. Nothing here knows a model.

Writes: at eviction, or earlier in the background past ``--spill-highwater``; a clean shutdown writes what is kept.
Copies off the device run on a writer's stream; the engine waits only before the rows move or go (``release``). Reads
run on a reader thread into pinned memory (``load_async``), so other streams keep decoding.

Files: safetensors, header padded to 4 KiB, O_DIRECT where it works, written to ``.partial``, fsynced, renamed. A file
whose size does not match its header is dropped, and every read checks a CRC-32 for each 8 MiB. One directory per
compatibility hash (format, build, runtime, weights, drafter, the engine's settings and cache layout) and per rank;
other hashes are pruned at start.

Several ranks: every decision is taken by all ranks in the same order from the index alone (never from how far this
rank's own writes got), so all hold the same index; a save agrees its size and the free-disk floor in one all-gather, a
read is used only when it worked on every rank, and at start only entries every rank holds are kept, in rank 0's
order.

Credits: wojo (Robert Wojciechowski, https://github.com/wojo) wrote this module and its hooks. The idea of NVMe session
state per rank was proposed in MiaAI-Lab's GLM-5.3-Flash vLLM kit pull request #232, authored by gabewillen (the idea
only, no code). The multi-rank design (one directory a rank, rank 0 deciding, an agreement on every read, a reconcile
at start) is that of patch 0250 of jayleaton/glm53-tensorfold-spark, authored by Jay Leaton (Apache-2.0). The file
conventions and flag names are those of TensorFold's MLX spill (pull request #68, authored by gilby) and of issue #155
(authored by raymondkpwong)."""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import hashlib
import importlib
import json
import os
import queue
import shutil
import struct
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

FORMAT = "tensorfold-cuda-spill-2"
CHUNK = 64 << 20                       # bytes a staging buffer
ALIGN = 4096                           # O_DIRECT offsets, lengths and buffers
BLOCK = 8 << 20                        # bytes one CRC-32 covers (CHUNK is a multiple of it)
STAGE = 8 << 20                        # bytes of the device buffer rows a memcpy may not reach pass through
WAIT_S = 60.0                          # a read of an entry still being written waits this long for it (reader thread)
INFLIGHT = 4 << 30                     # bytes of queued writes past which a new one first waits for the oldest

_DT = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64", torch.int8: "I8",
       torch.uint8: "U8", torch.int16: "I16", torch.int32: "I32", torch.int64: "I64", torch.bool: "BOOL"}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _name):
        _DT[getattr(torch, _name)] = _code
_TD = {v: k for k, v in _DT.items()}

_PINNED: dict[str, list[torch.Tensor]] = {}     # staging allocated before the engine sizes its caches


def _up(x: int) -> int:
    return -(-x // ALIGN) * ALIGN


@dataclass
class SpillConfig:
    """``tensorfold serve`` flags of the tier (every rank must be started with the same values)."""

    root: str
    gib: float
    highwater: float = 0.70            # memory occupancy past which states eviction would take next are written early
    min_tokens: int = 8192             # shorter prompts are not written
    min_free_gib: float = 50.0         # no write leaves less disk free (on any rank)
    keep_builds: int = 1
    writers: int = 2                   # writer threads (TF_SPILL_WRITERS), each with two staging buffers
    direct: bool = True                # O_DIRECT where the file system takes it (TF_SPILL_DIRECT=0: buffered)
    flush_s: float = 60.0              # a clean shutdown's budget (TF_SPILL_FLUSH_S)

    def __post_init__(self) -> None:
        if not 0.0 < self.highwater <= 1.0:
            raise ValueError(f"--spill-highwater: a fraction in (0, 1], not {self.highwater}")
        if self.gib <= 0:
            raise ValueError("--spill-gib must be above 0 to turn the tier on")
        if self.writers < 1:
            raise ValueError("TF_SPILL_WRITERS: at least 1")

    @classmethod
    def from_args(cls, args: Any) -> SpillConfig | None:
        gib = float(getattr(args, "spill_gib", 0.0) or 0.0)
        if gib <= 0:
            return None
        root = str(getattr(args, "snapshot_dir", "") or "")
        if not root or root.lower() == "none":
            raise ValueError("--spill-gib needs --snapshot-dir")
        return cls(root=os.path.expanduser(root), gib=gib,
                   highwater=float(getattr(args, "spill_highwater", 0.70)),
                   min_tokens=int(getattr(args, "spill_min_tokens", 8192)),
                   min_free_gib=float(getattr(args, "spill_min_free_gib", 50.0)),
                   keep_builds=int(getattr(args, "spill_keep_builds", 1)),
                   writers=int(os.environ.get("TF_SPILL_WRITERS", "2")),
                   direct=os.environ.get("TF_SPILL_DIRECT", "1") != "0",
                   flush_s=float(os.environ.get("TF_SPILL_FLUSH_S", "60")))


def _pinned(n: int) -> list[torch.Tensor]:
    pin = torch.cuda.is_available()
    return [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=pin) for _ in range(n)]


def _stage(device, stream) -> torch.Tensor:
    """A device buffer for ``_by_kernel``, allocated on ``stream`` (the one it is used on)."""

    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        return torch.empty(STAGE, dtype=torch.uint8, device=device)


def _by_kernel(dst: torch.Tensor, src: torch.Tensor, stage: torch.Tensor, stream) -> None:
    """``dst[:] = src`` (flat bytes) through ``stage`` a piece at a time, on ``stream``. For rows that a memcpy may not
    cross, only a kernel (patch 0072's display span: two host registrations, CUDA refuses a copy across them)."""

    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        for i in range(0, src.numel(), stage.numel()):
            k = min(stage.numel(), src.numel() - i)
            if src.device == stage.device:      # off the device: a kernel into the buffer, then a memcpy out
                torch.bitwise_or(src[i:i + k], 0, out=stage[:k])
                dst[i:i + k].copy_(stage[:k], non_blocking=stream is not None)
            else:                               # onto it: a memcpy into the buffer, then a kernel
                stage[:k].copy_(src[i:i + k], non_blocking=stream is not None)
                torch.bitwise_or(stage[:k], 0, out=dst[i:i + k])


def preallocate(cfg: SpillConfig | None) -> None:
    """Pin the staging buffers now, before the engine sizes its caches from the memory left (two a writer, two for
    reads), so the caches account for them."""

    if cfg is None or _PINNED:
        return
    _PINNED["write"] = _pinned(2 * cfg.writers)
    _PINNED["read"] = _pinned(2)


def build_id(model_dir, drafter_dir=None) -> str:
    """What stored states depend on beyond the cache layout, as MLX's prefix-snapshot key names it: the weights' (and
    drafter's) paths, the TensorFold version, the sources and kernels of the model's family and of tensorfold.cuda
    (read, not imported: the MLX kernel modules do not import on CUDA), and the runtime whose kernels make the bits
    (torch, CUDA, GPU, driver). Other values: another directory, nothing reused."""

    from pathlib import Path

    from tensorfold import __version__, families

    pkg = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    family = pkg.joinpath(*families.detect(model_dir).module.split(".")[1:])
    for base in (family, pkg / "cuda"):
        for path in sorted(p for p in base.rglob("*") if p.suffix in (".py", ".cu", ".cuh", ".cpp", ".h")):
            digest.update(path.relative_to(pkg).as_posix().encode())
            digest.update(path.read_bytes())
    run = f"torch={torch.__version__}|cuda={torch.version.cuda}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        run += f"|gpu={p.name}:{p.major}.{p.minor}"
    try:
        with open("/proc/driver/nvidia/version") as f:
            run += f"|driver={f.readline().strip()}"
    except OSError:
        pass
    drafter = f"|drafter={Path(drafter_dir).resolve()}" if drafter_dir is not None else ""
    return f"{Path(model_dir).resolve()}{drafter}|tensorfold={__version__}|sources={digest.hexdigest()[:16]}|{run}"


def weights_id(*dirs) -> str:
    """This rank's copy of the weights (and the drafter's): each file's name, size, mtime and inode (rewritten in
    place: a new folder)."""

    from pathlib import Path

    digest = hashlib.sha256()
    for d in (d for d in dirs if d is not None):
        for path in sorted(Path(d).iterdir()):
            if path.suffix in (".safetensors", ".json"):
                st = path.stat()                              # through the hub's links, to the files themselves
                digest.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}".encode())
    return digest.hexdigest()[:16]


def ids_key(ids: Sequence[int]) -> tuple[int, int]:
    h = hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).digest()
    return int.from_bytes(h[0:4], "big", signed=True), int.from_bytes(h[4:8], "big", signed=True)


def key_name(key: tuple[int, int], n: int) -> str:
    return f"{key[0] & 0xffffffff:08x}{key[1] & 0xffffffff:08x}_{n}"


# -- objects to files and back -----------------------------------------------------------------------------------------
def _enc(value: Any, name: str, items: list) -> dict:
    if isinstance(value, torch.Tensor):
        items.append((name, value))
        return {"t": name}
    if isinstance(value, np.ndarray):
        items.append((name, torch.from_numpy(np.ascontiguousarray(value))))
        return {"t": name, "np": 1}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"v": value}
    if isinstance(value, (list, tuple)):
        out = {"l": [_enc(v, f"{name}.{i}", items) for i, v in enumerate(value)]}
        if isinstance(value, tuple):
            out["tuple"] = 1
        return out
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"d": {k: _enc(v, f"{name}.{k}", items) for k, v in value.items()}}
    raise TypeError(f"the spill tier cannot store {type(value).__name__} at {name}")


def _dec(spec: dict, tensors: dict) -> Any:
    if "t" in spec:
        t = tensors[spec["t"]]
        return t.cpu().numpy() if spec.get("np") else t
    if "v" in spec:
        return spec["v"]
    if "l" in spec:
        out = [_dec(s, tensors) for s in spec["l"]]
        return tuple(out) if spec.get("tuple") else out
    if "d" in spec:
        return {k: _dec(s, tensors) for k, s in spec["d"].items()}
    raise ValueError(f"an unknown field spec {spec}")


def encode(obj: Any, *, skip: Sequence[str] = (),
           materialize: Callable[[Any], None] | None = None) -> tuple[list, dict]:
    """``obj``'s fields as (named tensors, a JSON-able layer): tensors by name, plain values, lists, tuples and string
    dicts of them; the class by import path. ``transient`` (class attribute) and ``skip`` name fields left out;
    ``materialize(obj)`` first (a family attaching rows that still live in its caches)."""

    if materialize is not None:
        materialize(obj)
    cls = type(obj)
    leave = set(getattr(cls, "transient", ())) | set(skip)
    items: list = []
    fields = {k: _enc(v, f"0.{k}", items) for k, v in vars(obj).items() if k not in leave and not k.startswith("_")}
    return items, {"class": f"{cls.__module__}:{cls.__qualname__}", "fields": fields}


def decode(layer: dict, tensors: dict, classes: Sequence[str]) -> Any:
    """The object ``encode`` described (a dataclass's missing fields get their defaults); ``classes``: the only
    ``module:qualname`` paths it may build (nothing else is imported)."""

    if layer["class"] not in classes:
        raise ValueError(f"a stored state of class {layer['class']!r}, not one of {list(classes)}")
    mod, _, qual = layer["class"].partition(":")
    cls: Any = importlib.import_module(mod)
    for part in qual.split("."):
        cls = getattr(cls, part)
    obj = cls.__new__(cls)
    for k, spec in layer["fields"].items():
        setattr(obj, k, _dec(spec, tensors))
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if not hasattr(obj, f.name):
                if f.default is not dataclasses.MISSING:
                    setattr(obj, f.name, f.default)
                elif f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
                    setattr(obj, f.name, f.default_factory())             # type: ignore[misc]
    return obj


def _dtype_code(t: torch.Tensor) -> str:
    try:
        return _DT[t.dtype]
    except KeyError:
        raise TypeError(f"the spill tier cannot store {t.dtype}") from None


def _crc_placeholder(nbytes: int) -> str:
    """The header's CRC field before the data is hashed: one 8-digit slot a BLOCK, so the header keeps its length."""

    return ",".join(["00000000"] * (-(-nbytes // BLOCK)))


class _Crcs:
    """CRC-32 of each BLOCK of a byte stream fed in order."""

    def __init__(self) -> None:
        self.done: list[int] = []
        self.cur = 0
        self.fill = 0

    def feed(self, mv) -> None:
        while len(mv):
            k = min(BLOCK - self.fill, len(mv))
            self.cur = zlib.crc32(mv[:k], self.cur)
            self.fill += k
            mv = mv[k:]
            if self.fill == BLOCK:
                self.done.append(self.cur)
                self.cur, self.fill = 0, 0

    def hex(self) -> str:
        return ",".join(f"{c:08x}" for c in self.done + ([self.cur] if self.fill else []))


def _header(items: list, meta: dict) -> tuple[bytes, int]:
    """The safetensors header for ``items`` padded so the data starts on a 4 KiB boundary; (bytes incl. its 8-byte
    length, data bytes)."""

    table: dict = {}
    at = 0
    for name, t in items:
        nbytes = t.numel() * t.element_size()
        table[name] = {"dtype": _dtype_code(t), "shape": list(t.shape), "data_offsets": [at, at + nbytes]}
        at += nbytes
    body = json.dumps({"__metadata__": {k: str(v) for k, v in meta.items()}, **table}, separators=(",", ":")).encode()
    total = _up(8 + len(body))
    body += b" " * (total - 8 - len(body))
    return struct.pack("<Q", len(body)) + body, at


@dataclass
class Entry:
    key: tuple[int, int]
    n: int
    name: str
    size: int = 0                      # exact bytes of the file, agreed by every rank (the largest)
    last: int = 0
    ids: Any = None                    # np.int64, keyed ids for a prompt with pictures (rank 0 matches against it)
    head: bool = False
    fields: dict = field(default_factory=dict)       # the stored object's plain fields: a family's test before a read
    bad: bool = False                  # this rank failed to write it (a read fails here; every rank then forgets it)


@dataclass
class Stats:
    written: int = 0
    write_failed: int = 0
    skipped: int = 0
    full: int = 0
    no_space: int = 0
    deferred: int = 0                  # early writes put off (the writers were busy); the engine counts them
    behind: int = 0                    # early writes queued
    unwritten_evicted: int = 0         # states that left memory without being written (too short, off grid, ...)
    unwritten: dict = field(default_factory=dict)     # unwritten_evicted by reason (/health: unwritten_<reason>)
    loaded: int = 0
    load_failed: int = 0
    crc_failed: int = 0                # reads whose data did not match its CRC-32 (the entry goes, a prefill)
    load_unfit: int = 0                # reads without the draft caches the request needs (the entry goes, a prefill)
    pruned: int = 0
    save_s: float = 0.0
    write_s: float = 0.0
    load_s: float = 0.0
    copy_wait_s: float = 0.0           # the engine waited for a copy off the device before moving rows
    copy_waits: int = 0
    behind_s: float = 0.0              # the engine choosing early writes (each iteration past the high-water mark)
    written_bytes: int = 0
    loaded_bytes: int = 0
    loaded_tokens: int = 0
    restore_ms: list = field(default_factory=list)


class Job:
    """One entry on its way to disk. ``copied``: every byte left the device (its rows may move); ``done``: on disk or
    given up."""

    def __init__(self, entry: Entry, path: str, header: bytes, items: list, src_event, seq: int = 0) -> None:
        self.entry = entry
        self.key = entry.key
        self.path = path
        self.partial = f"{path}.{seq}.partial"      # its own: a later write of the same prompt never shares it
        self.nbytes = entry.size
        self.header = header
        self.items = items
        self.src_event = src_event
        self.copied = threading.Event()
        self.done = threading.Event()
        self.ok = False


class LoadJob:
    """A read on the reader thread: ``done`` once every byte is on the device (``event`` recorded on the reader's
    stream) or the read failed (``error``)."""

    def __init__(self, entry: Entry, path: str, header: dict, data_at: int, dests: dict) -> None:
        self.entry = entry
        self.path = path
        self.header = header
        self.data_at = data_at
        self.dests = dests             # name -> device tensor the bytes go to
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.event = None
        self.writing: Job | None = None           # the entry's write, still running when the read was asked for
        self.t0 = time.perf_counter()
        self.meta = header.get("__metadata__", {}) or {}


def _read_header_file(f) -> tuple[dict, int]:
    (size,) = struct.unpack("<Q", f.read(8))
    if size > 1 << 30:
        raise ValueError("not a spill file")
    return json.loads(f.read(size)), 8 + size


def _parse_header(raw: bytes) -> tuple[dict, int]:
    (size,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + size]), 8 + size


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % ALIGN == 0


def _plain(layers: list) -> dict:
    """A stored object's plain fields (numbers, strings, None) from its JSON layer."""

    return {k: s["v"] for k, s in layers[0]["fields"].items() if "v" in s} if layers else {}


class SpillStore:
    """The tier's files on this rank: an index every rank holds alike, writer threads, a reader thread."""

    def __init__(self, cfg: SpillConfig, *, rank: int = 0, world: int = 1, device: Any = "cuda", signature: str = "",
                 model_id: str = "", share: Callable[[list[int] | None], list[int]] | None = None,
                 gather: Callable[[list[int]], list[list[int]]] | None = None, quiet: bool = False,
                 weights: str = "", classes: Sequence[str] = (), spanned: Sequence[str] = ()) -> None:
        self.cfg = cfg
        self.classes = classes                  # the snapshot classes a read may build (decode)
        self.spanned = frozenset(f"rows.{n}" for n in spanned)   # rows (``save``'s names) a memcpy may not reach:
                                                                 # moved by a kernel (``_by_kernel``)
        self.rank, self.world = int(rank), int(world)
        self.device = device
        self.model_id = model_id
        self.share, self.gather = share, gather
        self.cap = int(cfg.gib * (1 << 30))
        shared = hashlib.sha256(f"{FORMAT}|{signature}".encode()).digest()
        own = hashlib.sha256(weights.encode()).digest()       # this rank's copy of the weights
        words = [int.from_bytes(h[i:i + 4], "big", signed=True) for h in (shared, own) for i in range(0, 16, 4)]
        if self.world > 1 and gather is not None:             # one directory name on every rank
            rows = gather(words)
            differ = [r for r, row in enumerate(rows) if list(row[:4]) != list(rows[0][:4])]
            if differ:
                print(f"[tensorfold] spill rank {self.rank}: rank(s) {differ} run another build or settings than "
                      "rank 0, so no rank can resume another's stored prompts until they match", flush=True)
            words = [w for row in rows for w in row[:8]]
        self.compat = hashlib.sha256(np.asarray(words, dtype=np.int32).tobytes()).hexdigest()[:16]
        self.top = os.path.join(cfg.root, f"spill-{self.compat}")
        self.dir = os.path.join(self.top, f"rank{self.rank}")
        try:                                                  # files belong to --snapshot-dir's owner: a server run as
            st = os.stat(cfg.root)                            # root in a container writes them as you
            self.uid, self.gid = st.st_uid, st.st_gid
        except OSError:
            self.uid = self.gid = -1
        self.lock = threading.RLock()
        self.index: dict[tuple, Entry] = {}
        self.pending: dict[tuple, Job] = {}
        self.clock = 0
        self.seq = 0
        self.generation = 0                     # bumped whenever the index changes
        self.stats = Stats()
        self.direct = bool(cfg.direct) and hasattr(os, "O_DIRECT")
        self.jobs: queue.Queue = queue.Queue()
        self.reads: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._reader: threading.Thread | None = None
        self._dirty = False
        self._last_dir_sync = 0.0
        self._dir_lock = threading.Lock()
        problem, pruned_builds = "", 0
        try:
            for d in (self.top, self.dir):                    # prompts' token ids: this user's only
                os.makedirs(d, mode=0o700, exist_ok=True)
                os.chmod(d, 0o700)
                self._own(d)
            pruned_builds = self._prune_builds()
            self._scan()
        except OSError as exc:
            problem = repr(exc)
        flags = gather([int(bool(problem))]) if self.world > 1 and gather is not None else [[int(bool(problem))]]
        bad = [r for r, row in enumerate(flags) if row[0]]
        if bad:                                               # every rank stops alike: none waits on another
            raise RuntimeError(f"--snapshot-dir {cfg.root}: rank(s) {bad} cannot use it"
                               + (f": {problem}" if problem else ""))
        found = len(self.index)
        dropped = self._reconcile() if self.world > 1 else 0
        if self.rank == 0 and not quiet:
            print(f"[tensorfold] spill: {self.top} ({cfg.gib:g} GiB a rank, early writes past "
                  f"{cfg.highwater:.0%} of memory, from {cfg.min_tokens} tokens, {cfg.writers} writers"
                  f"{', O_DIRECT' if self.direct else ''}, floor {cfg.min_free_gib:g} GiB free): "
                  f"{found - dropped} stored prompts from before this start, {self._total() / (1 << 30):.1f} GiB"
                  f"{f'; pruned {pruned_builds} older builds' if pruned_builds else ''}", flush=True)

    # -- housekeeping --------------------------------------------------------------------------------------------------
    def _own(self, path: str) -> None:
        if self.uid >= 0:
            try:
                os.chown(path, self.uid, self.gid)
            except OSError:
                pass

    def _prune_builds(self) -> int:
        """Folders of other compatibility hashes (other builds, models, layouts): the newest ``keep_builds`` stay."""

        try:
            others = [os.path.join(self.cfg.root, d) for d in os.listdir(self.cfg.root)
                      if d.startswith("spill-") and os.path.join(self.cfg.root, d) != self.top]
        except OSError:
            return 0
        others = [d for d in others if os.path.isdir(d)]
        others.sort(key=lambda d: os.stat(d).st_mtime, reverse=True)
        gone = 0
        for d in others[max(0, self.cfg.keep_builds):]:
            mine = os.path.join(d, f"rank{self.rank}")
            if os.path.isdir(mine):
                shutil.rmtree(mine, ignore_errors=True)
                gone += 1
            try:
                os.rmdir(d)                                    # empty once every rank pruned its own
            except OSError:
                pass
        return gone

    def _total(self) -> int:
        with self.lock:
            return sum(e.size for e in self.index.values())

    def disk_free(self) -> int:
        try:
            st = os.statvfs(self.dir)
            return st.f_bavail * st.f_frsize
        except OSError:
            return 0

    # -- the index -----------------------------------------------------------------------------------------------------
    def _scan(self) -> None:
        found = []
        for name in os.listdir(self.dir):
            path = os.path.join(self.dir, name)
            if name.endswith(".partial"):                      # a write cut short: never read
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue
            if name.endswith(".safetensors"):
                try:
                    found.append((os.stat(path).st_mtime, name))
                except OSError:
                    pass
        for _, name in sorted(found):                          # oldest first: recency follows the files' age
            path = os.path.join(self.dir, name)
            try:
                with open(path, "rb") as f:
                    header, data_at = _read_header_file(f)
                    meta = header.get("__metadata__") or {}
                    if meta.get("format") != FORMAT or meta.get("compat") != self.compat:
                        raise ValueError("another format")
                    want = data_at + max([v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
                                         or [0])
                    if os.fstat(f.fileno()).st_size != want:
                        raise ValueError("size does not match the header")
                ids = np.asarray(json.loads(meta.get("keyed") or meta["tokens"]), dtype=np.int64)
                n = int(meta["n"])
                key = ids_key(ids)
                if len(ids) != n or key_name(key, n) + ".safetensors" != name:
                    raise ValueError("tokens do not match the name")
                e = Entry(key, n, name[:-12], want, ids=ids, head=meta.get("head") == "1",
                          fields=_plain(json.loads(meta.get("layers", "[]"))))
                self.clock += 1
                e.last = self.clock
                self.index[key] = e
            except Exception:                                  # noqa: BLE001  (unreadable, cut or another format: gone)
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _reconcile(self) -> int:
        """Every rank: keep the entries every rank holds, in rank 0's recency order, at the largest rank's size.
        Returns how many this rank dropped."""

        if self.share is None or self.gather is None:
            return 0
        mine = sorted(self.index.values(), key=lambda e: e.last)
        flat = [v for e in mine for v in (e.key[0], e.key[1], e.n)] if self.rank == 0 else None
        theirs = self.share(flat)
        keys = [(theirs[i], theirs[i + 1], theirs[i + 2]) for i in range(0, len(theirs), 3)]
        keep = []
        if keys:
            held = [int((hi, lo) in self.index and self.index[(hi, lo)].n == n) for hi, lo, n in keys]
            every = self.gather(held)
            keep = [k for i, k in enumerate(keys) if all(r[i] for r in every)]
        sizes = self.gather([self.index[(hi, lo)].size >> 20 for hi, lo, _ in keep]) if keep else []
        want = {(hi, lo) for hi, lo, _ in keep}
        dropped = 0
        for e in list(self.index.values()):
            if e.key not in want:
                self._delete(e)
                dropped += 1
        self.clock = 0
        for i, (hi, lo, _) in enumerate(keep):
            self.clock += 1
            e = self.index[(hi, lo)]
            e.last = self.clock
            e.size = max(e.size, (max(r[i] for r in sizes) + 1) << 20)
        return dropped

    def _delete(self, e: Entry) -> None:
        with self.lock:
            if self.index.get(e.key) is e:
                self.index.pop(e.key, None)
                self.generation += 1
            job = self.pending.get(e.key)
            if job is None or job.entry is not e:             # (one being written: its writer drops the file)
                try:
                    os.remove(os.path.join(self.dir, e.name + ".safetensors"))
                except OSError:
                    pass
            self.stats.pruned += 1

    def touch(self, e: Entry) -> None:
        with self.lock:
            self.clock += 1
            e.last = self.clock

    def touch_key(self, key: tuple) -> None:
        """Every rank alike: entry ``key`` becomes the most recently used (the clock moves even when it is gone)."""

        with self.lock:
            self.clock += 1
            e = self.index.get(key)
            if e is not None:
                e.last = self.clock

    def forget(self, key: tuple) -> None:
        """Every rank: an entry that failed to read goes (its files are not trusted)."""

        with self.lock:
            e = self.index.get(key)
        if e is not None:
            self._delete(e)

    def has(self, ids: Sequence[int], key: tuple[int, int] | None = None) -> bool:
        """``ids`` is stored (``key``: its ``ids_key``, if known)."""

        key = ids_key(ids) if key is None else key
        with self.lock:
            e = self.index.get(key)
            return e is not None and e.n == len(ids)      # rank-local states (bad, pending) never decide

    def pending_jobs(self) -> int:
        with self.lock:
            return len(self.pending)

    def find(self, prompt: Sequence[int], *, grid: int = 0, whole_needs_head: bool = True,
             usable: Callable[[Entry], bool] | None = None) -> Entry | None:
        """The longest stored prompt that ``prompt`` extends (or equals, when its head row was stored)."""

        arr = np.asarray(prompt, dtype=np.int64)
        best = None
        with self.lock:
            entries = list(self.index.values())
        for e in entries:
            if e.ids is None or e.bad or e.n > len(arr) or (e.n == len(arr) and whole_needs_head and not e.head):
                continue
            if (best is not None and e.n <= best.n) or (grid and e.n % grid):
                continue
            if usable is not None and not usable(e):
                continue
            if np.array_equal(arr[:e.n], e.ids):
                best = e
        return best

    def _make_room(self, extra: int, keep: frozenset = frozenset(), cap: int | None = None) -> bool:
        """Every rank alike: the least recently used entries go until ``extra`` more bytes fit the cap (or ``cap``, when
        the free-disk floor asks for less). The index alone
        decides (an entry still being written here goes too, its writer then drops the file), so every rank drops the
        same entries whatever its own writes' progress. ``keep``'s entries stay (an early write's: copies of prompts
        still in memory, which would otherwise evict each other in turn under a small cap)."""

        with self.lock:
            cap = self.cap if cap is None else min(cap, self.cap)
            total = self._total() + extra
            if total > cap and extra + sum(e.size for e in self.index.values() if e.key in keep) > cap:
                return False                       # it cannot fit beside ``keep``'s entries: nothing goes for it
            while total > cap:
                old = [e for e in self.index.values() if e.key not in keep]
                if not old:
                    break
                victim = min(old, key=lambda e: (e.last, e.key))
                total -= victim.size
                self._delete(victim)
            return total <= cap

    def _throttle(self, size: int) -> None:
        """A queued write holds its tensors (from live rows, the caller's caches too) until it is on disk: past
        ``INFLIGHT`` bytes queued, this rank first waits for its oldest write (only a wait: it decides nothing)."""

        while True:
            with self.lock:
                jobs = [j for j in self.pending.values() if not j.done.is_set()]
            held = sum(j.nbytes for j in jobs)
            if not jobs or held + size <= INFLIGHT:
                return
            jobs[0].done.wait()

    # -- writing -------------------------------------------------------------------------------------------------------
    def _start(self) -> None:
        if self._threads:
            return
        bufs = _PINNED.get("write") or _pinned(2 * self.cfg.writers)
        if any(not _aligned(b) for b in bufs):
            self.direct = False
        for i in range(self.cfg.writers):
            t = threading.Thread(target=self._writer, args=(bufs[2 * i:2 * i + 2],),
                                 name=f"spill-writer-{self.rank}-{i}",
                                 daemon=True)
            t.start()
            self._threads.append(t)

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every queued write is on disk (a shutdown). True when the writes finished in time."""

        end = None if timeout is None else time.monotonic() + timeout
        while True:
            with self.lock:
                jobs = list(self.pending.values())
            if not jobs:
                break
            left = None if end is None else max(0.0, end - time.monotonic())
            if not jobs[0].done.wait(left) and left is not None and left <= 0:
                return False
        self._sync_dirs(force=True)
        return True

    def _sync_dirs(self, force: bool = False) -> None:
        with self._dir_lock:
            now = time.monotonic()
            if self._dirty and (force or now - self._last_dir_sync >= 1.0):
                _fsync_dir(self.dir)
                self._dirty = False
                self._last_dir_sync = now

    def _open_write(self, path: str) -> tuple[int, bool]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if self.direct:
            try:
                return os.open(path, flags | os.O_DIRECT, 0o600), True
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                self.direct = False
        return os.open(path, flags, 0o600), False

    @staticmethod
    def _write_all(fd: int, mv: memoryview) -> None:
        while len(mv):
            k = os.write(fd, mv)
            mv = mv[k:]

    @staticmethod
    def _pwrite_all(fd: int, mv: memoryview, at: int) -> None:
        while len(mv):
            k = os.pwrite(fd, mv, at)
            mv, at = mv[k:], at + k

    def _write(self, job: Job, bufs: list[torch.Tensor], stream, hasher: ThreadPoolExecutor) -> bool:
        """One entry: header, then every tensor's bytes packed into the two staging buffers in turn (a buffer's copy
        off the device overlaps the other's write), full aligned buffers written, the tail padded then truncated."""

        path = job.partial
        fd, direct = self._open_write(path)
        total = 0
        try:
            if stream is not None and job.src_event is not None:
                stream.wait_event(job.src_event)        # the rows are final where the engine recorded
            cur, fill, ev = 0, 0, [None, None]
            hdr = np.frombuffer(job.header, dtype=np.uint8)
            bufs[cur][:len(hdr)].copy_(torch.from_numpy(hdr.copy()))
            fill = len(hdr)
            pending = None                              # (buffer index, bytes) copied, not yet written
            crcs, skip = _Crcs(), [len(hdr)]            # the data's CRCs (the header in the first buffer is not data)

            def flush(i: int, n: int) -> None:
                nonlocal total
                if ev[i] is not None:
                    ev[i].synchronize()
                    ev[i] = None
                k = _up(n) if direct else n
                mv = memoryview(bufs[i].numpy())
                hashed = hasher.submit(crcs.feed, mv[skip[0]:n])     # beside the write, which reads the same bytes
                skip[0] = 0
                self._write_all(fd, mv[:k])
                hashed.result()
                total += n

            stage = None
            for name, t in job.items:
                v = t.reshape(-1).view(torch.uint8)
                if name in self.spanned and stage is None:
                    stage = _stage(v.device, stream)
                s, size = 0, v.numel()
                while s < size:
                    k = min(CHUNK - fill, size - s)
                    if name in self.spanned:
                        _by_kernel(bufs[cur][fill:fill + k], v[s:s + k], stage, stream)
                    elif stream is not None and v.is_cuda:
                        with torch.cuda.stream(stream):
                            bufs[cur][fill:fill + k].copy_(v[s:s + k], non_blocking=True)
                    else:
                        bufs[cur][fill:fill + k].copy_(v[s:s + k])
                    fill += k
                    s += k
                    if fill == CHUNK:
                        if stream is not None:
                            e = torch.cuda.Event()
                            e.record(stream)
                            ev[cur] = e
                        if pending is not None:
                            flush(*pending)
                        pending = (cur, CHUNK)
                        cur ^= 1
                        if ev[cur] is not None:
                            ev[cur].synchronize()
                            ev[cur] = None
                        fill = 0
            if stream is not None and fill:
                e = torch.cuda.Event()
                e.record(stream)
                ev[cur] = e
            for i in (0, 1):
                if ev[i] is not None:
                    ev[i].synchronize()
            job.copied.set()                            # every byte is in host memory: the rows may move now
            if pending is not None:
                flush(*pending)
            if fill:
                flush(cur, fill)
            slot = b'"crc32":"' + _crc_placeholder(len(crcs.done) * BLOCK + crcs.fill).encode() + b'"'
            final = job.header.replace(slot, b'"crc32":"' + crcs.hex().encode() + b'"')
            if job.header.count(slot) != 1 or len(final) != len(job.header):
                raise RuntimeError("the header's CRC field is missing or changed length")
            head = np.frombuffer(final, dtype=np.uint8)
            bufs[0][:len(head)].copy_(torch.from_numpy(head.copy()))
            self._pwrite_all(fd, memoryview(bufs[0].numpy())[:len(head)], 0)
            os.ftruncate(fd, len(job.header) + sum(t.numel() * t.element_size() for _, t in job.items))
            os.fsync(fd)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            os.close(fd)
            fd = -1
            with self.lock:                             # still this prompt's entry: its file (one dropped meanwhile,
                current = self.index.get(job.key) is job.entry       # by the cap or a failed read, leaves none)
                if current:
                    os.replace(path, job.path)
            if not current:
                os.remove(path)
                return False
            self._own(job.path)
            with self._dir_lock:
                self._dirty = True
            self._sync_dirs()
            with self.lock:
                self.stats.written_bytes += total
            return True
        except Exception as exc:                         # noqa: BLE001  (only this prompt is not written)
            self._log(exc)
            job.copied.set()
            try:
                if fd >= 0:
                    os.close(fd)
            except OSError:
                pass
            try:
                os.remove(path)
            except OSError:
                pass
            return False

    def _writer(self, bufs: list[torch.Tensor]) -> None:
        stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
        hasher = ThreadPoolExecutor(1, thread_name_prefix=f"spill-crc-w{self.rank}")
        while True:
            job = self.jobs.get()
            t0 = time.perf_counter()
            try:
                ok = self._write(job, bufs, stream, hasher)
            except Exception as exc:                   # noqa: BLE001  (the writer never dies)
                self._log(exc)
                ok = False
            job.ok = ok
            self._entry_done(job, ok)
            job.items = None                            # its tensors go now, not when the next write arrives
            job.copied.set()
            job.done.set()
            with self.lock:
                self.stats.write_s += time.perf_counter() - t0
            job = None

    def _log(self, exc: BaseException) -> None:
        with self.lock:
            self.stats.write_failed += 1
        print(f"[tensorfold] spill rank {self.rank}: write failed: {exc!r}", flush=True)

    def save(self, ids: Sequence[int], obj: Any = None, rows: Sequence[tuple[str, torch.Tensor]] = (), *,
             skip: Sequence[str] = (), materialize: Callable[[Any], None] | None = None, head: bool = False,
             meta: dict | None = None,
             tokens: Sequence[int] | None = None, keep: frozenset = frozenset()) -> Job | None:
        """Every rank, in the same order: queue prompt ``ids``'s state (``obj``'s fields, see ``encode``) and the
        named ``rows`` (read back into caller views) for writing; one all-gather agrees the size and the free-disk
        floor. Returns the job, or None when skipped (stored or on its way, too short, no room, no disk)."""

        t0 = time.perf_counter()
        n = len(ids)
        if n < self.cfg.min_tokens:
            return None
        key = ids_key(ids)
        with self.lock:
            e = self.index.get(key)
            if e is not None and e.n == n:             # stored or on its way (every rank alike)
                self.touch(e)
                self.stats.skipped += 1
                return None
        items, layer = encode(obj, skip=skip, materialize=materialize) if obj is not None else ([], None)
        items = [(k, t if t.is_contiguous() else t.contiguous()) for k, t in items]
        for name, t in rows:
            if not t.is_contiguous():
                raise ValueError(f"rows {name}: not contiguous")
            items.append((f"rows.{name}", t))
        info = {"format": FORMAT, "compat": self.compat, "model": self.model_id, "n": n,
                "tokens": json.dumps([int(t) for t in (ids if tokens is None else tokens)]),
                **({} if tokens is None else {"keyed": json.dumps([int(t) for t in ids])}),
                "layers": json.dumps([layer] if layer else []),
                "rows": json.dumps([nm for nm, _ in rows]), "head": "1" if head else "0",
                "saved": f"{time.time():.3f}",
                "crc32": _crc_placeholder(sum(t.numel() * t.element_size() for _, t in items)), **(meta or {})}
        header, data = _header(items, info)
        size = len(header) + data
        floor = int(self.cfg.min_free_gib * (1 << 30))
        with self.lock:                            # (writes still queued here count as written)
            queued = sum(j.nbytes for j in self.pending.values() if not j.done.is_set())
        short = max(0, floor + size + queued - self.disk_free())   # bytes this rank's disk would be below the floor
        if self.world > 1 and self.gather is not None:
            got = self.gather([-(-short >> 20), (size >> 20) + 1])
            short = max(r[0] for r in got) << 20
            size = max(r[1] for r in got) << 20
        if short and not self._make_room(size, keep, cap=self._total() + size - short):
            self.stats.no_space += 1               # below the floor even with every entry outside ``keep`` gone
            return None
        if size > self.cap or not self._make_room(size, keep):   # (larger than the cap: nothing is evicted)
            self.stats.full += 1
            return None
        self._start()
        self._throttle(size)
        name = key_name(key, n)
        entry = Entry(key, n, name, size, ids=np.asarray(ids, dtype=np.int64), head=head,
                      fields=_plain([layer] if layer else []))
        src = None
        if torch.cuda.is_available():
            src = torch.cuda.Event()
            src.record(torch.cuda.current_stream())
        with self.lock:
            self.seq += 1
            seq = self.seq
        job = Job(entry, os.path.join(self.dir, name + ".safetensors"), header, items, src, seq)
        with self.lock:                                    # the index changes here, on every rank alike
            old = self.index.get(key)
            if old is not None:
                self.index.pop(key, None)
            self.pending[key] = job
            self.clock += 1
            entry.last = self.clock                        # recency is the queue order (every rank alike)
            self.index[key] = entry
            self.generation += 1
        self.jobs.put(job)
        self.stats.save_s += time.perf_counter() - t0
        return job

    def release(self, job: Job) -> None:
        """The engine is about to move or free the rows ``job`` reads: wait until they are off the device."""

        if job.copied.is_set():
            return
        w0 = time.perf_counter()
        job.copied.wait()
        self.stats.copy_wait_s += time.perf_counter() - w0
        self.stats.copy_waits += 1

    def _entry_done(self, job: Job, ok: bool) -> None:
        with self.lock:
            if self.pending.get(job.key) is job:
                self.pending.pop(job.key, None)
            if ok:
                self.stats.written += 1
                if job.key not in self.index:              # dropped after its rename: the file goes too
                    try:
                        os.remove(job.path)
                    except OSError:
                        pass
            elif self.index.get(job.key) is job.entry:
                job.entry.bad = True                       # stays indexed (every rank alike); a read fails here

    # -- reading -------------------------------------------------------------------------------------------------------
    def load_async(self, key: tuple, dst_for: Callable[[str, list[int], torch.dtype], torch.Tensor | None]) -> LoadJob:
        """Start reading entry ``key`` on the reader thread. ``dst_for(name, shape, dtype)`` places a tensor (a view the
        caller owns) or returns None for a new one, allocated here on the calling thread's stream. Raises at once when
        this rank has no readable entry; read errors land in the job. An entry still being written is read once its
        write ends, on the reader thread (this one does not wait)."""

        with self.lock:
            e = self.index.get(key)
            job = self.pending.get(key)
        if e is None or e.bad:
            raise FileNotFoundError(f"rank {self.rank} has no readable stored prompt {key}")
        path = os.path.join(self.dir, e.name + ".safetensors")
        writing = job if job is not None and job.entry is e and not job.done.is_set() else None
        if writing is not None:                        # its header is at hand; its CRCs come with the file (``_read``)
            header, data_at = _parse_header(writing.header)
        else:
            with open(path, "rb") as f:
                header, data_at = _read_header_file(f)
        meta = header.get("__metadata__") or {}
        if meta.get("format") != FORMAT or meta.get("compat") != self.compat:
            raise ValueError(f"{path}: another format or build")
        dests = {}
        for name, d in header.items():
            if name == "__metadata__":
                continue
            dt, shape = _TD[d["dtype"]], d["shape"]
            dst = dst_for(name, shape, dt)
            if dst is None:
                dst = torch.empty(shape, dtype=dt, device=self.device)
            elif list(dst.shape) != list(shape) or dst.dtype != dt:
                raise ValueError(f"{name}: file {shape} {dt}, destination {list(dst.shape)} {dst.dtype}")
            if not dst.is_contiguous():
                raise ValueError(f"{name}: the destination is not contiguous")
            dests[name] = dst
        lj = LoadJob(e, path, header, data_at, dests)
        lj.writing = writing
        if torch.cuda.is_available():
            lj.after = torch.cuda.Event()
            lj.after.record(torch.cuda.current_stream())      # destinations are free to write from here
        else:
            lj.after = None
        if self._reader is None:
            self._reader = threading.Thread(target=self._read_loop, name=f"spill-reader-{self.rank}", daemon=True)
            self._reader.start()
        self.reads.put(lj)
        return lj

    def _read_loop(self) -> None:
        pins = _PINNED.get("read") or _pinned(2)
        stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
        pool = ThreadPoolExecutor(4, thread_name_prefix=f"spill-crc-r{self.rank}")
        while True:
            lj = self.reads.get()
            try:
                self._read(lj, pins, stream, pool)
            except BaseException as exc:                   # noqa: BLE001  (the reader never dies)
                lj.error = exc
            lj.done.set()
            lj = None                                      # (the caller holds what it read, not this thread)

    def _read(self, lj: LoadJob, pins: list[torch.Tensor], stream, pool: ThreadPoolExecutor) -> None:
        w, lj.writing = lj.writing, None
        if w is not None:                               # the entry's write first; the file's header has its CRCs
            if not w.done.wait(WAIT_S) or not w.ok:
                raise OSError(f"{lj.path}: its write failed or did not end in time")
            with open(lj.path, "rb") as f:
                header, data_at = _read_header_file(f)
            table = {k: v for k, v in header.items() if k != "__metadata__"}
            if data_at != lj.data_at or table != {k: v for k, v in lj.header.items() if k != "__metadata__"}:
                raise ValueError(f"{lj.path}: not the file that was being written")
            lj.header, lj.meta = header, header.get("__metadata__") or {}
        direct = self.direct and all(_aligned(p) for p in pins) and lj.data_at % ALIGN == 0
        flags = os.O_RDONLY
        fd = -1
        if direct:
            try:
                fd = os.open(lj.path, flags | os.O_DIRECT)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                direct = False
        if fd < 0:
            fd = os.open(lj.path, flags)
        try:
            if stream is not None and lj.after is not None:
                stream.wait_event(lj.after)
            table = sorted(((k, v) for k, v in lj.header.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            end = max([v["data_offsets"][1] for _, v in table] or [0])
            if os.fstat(fd).st_size != lj.data_at + end:
                raise EOFError(f"{lj.path}: the file is shorter or longer than its header says")
            views = [(v["data_offsets"][0], v["data_offsets"][1], lj.dests[k].reshape(-1).view(torch.uint8),
                      k in self.spanned) for k, v in table]
            stage = next((_stage(v.device, stream) for _, _, v, spanned in views if spanned), None)
            ev = [None, None]
            hashing: list = [[], []]                    # a pin's block hashes in flight (they read the pin)
            sums: list = []
            flip, s, ti = 0, 0, 0
            while s < end:
                k = min(CHUNK, end - s)
                pin = pins[flip]
                if ev[flip] is not None:
                    ev[flip].synchronize()
                for h in hashing[flip]:
                    h.result()
                want = _up(k) if direct else k
                mv = memoryview(pin.numpy())[:want]
                at = lj.data_at + s
                got = os.preadv(fd, [mv], at) if hasattr(os, "preadv") else os.pread(fd, want, at)
                if isinstance(got, bytes):
                    mv[:len(got)] = got
                    got = len(got)
                if got < k:
                    raise EOFError(f"{lj.path}: short read")
                hashing[flip] = [pool.submit(zlib.crc32, mv[i:min(i + BLOCK, k)]) for i in range(0, k, BLOCK)]
                sums += hashing[flip]
                lo, hi = s, s + k
                while ti < len(views) and views[ti][1] <= lo:
                    ti += 1
                j = ti
                ctx = torch.cuda.stream(stream) if stream is not None else None
                if ctx is not None:
                    ctx.__enter__()
                try:
                    while j < len(views) and views[j][0] < hi:
                        a, b, v, spanned = views[j]
                        x0, x1 = max(a, lo), min(b, hi)
                        if x1 > x0 and spanned:
                            _by_kernel(v[x0 - a:x1 - a], pin[x0 - lo:x1 - lo], stage, stream)
                        elif x1 > x0:
                            v[x0 - a:x1 - a].copy_(pin[x0 - lo:x1 - lo], non_blocking=stream is not None)
                        j += 1
                finally:
                    if ctx is not None:
                        ctx.__exit__(None, None, None)
                if stream is not None:
                    e = torch.cuda.Event()
                    e.record(stream)
                    ev[flip] = e
                flip ^= 1
                s = hi
                self.stats.loaded_bytes += k
            if stream is not None:
                e = torch.cuda.Event()
                e.record(stream)
                lj.event = e
                e.synchronize()
            stored = [int(c, 16) for c in str(lj.meta.get("crc32", "")).split(",") if c]
            seen = [h.result() for h in sums]
            if seen != stored:
                bad = next((i for i, (a, b) in enumerate(zip(seen, stored)) if a != b), min(len(seen), len(stored)))
                self.stats.crc_failed += 1
                raise ValueError(f"{lj.path}: block {bad} does not match its CRC-32")
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        except BaseException:
            if stream is not None:                      # copies still queued write rows that may move or go now
                stream.synchronize()
            raise
        finally:
            os.close(fd)

    def finish_load(self, lj: LoadJob, *, wait: bool = True) -> tuple[Any, dict, dict]:
        """After ``lj`` is done on this rank: (the decoded object or None, its row tensors by name, metadata). The
        caller's stream waits for the copies. Raises the read's error. Counts the restore."""

        if wait:
            lj.done.wait()
        if lj.error is not None:
            raise lj.error
        if lj.event is not None and torch.cuda.is_available():
            torch.cuda.current_stream().wait_event(lj.event)
        meta = lj.meta
        layers = json.loads(meta.get("layers", "[]"))
        obj = decode(layers[0], lj.dests, self.classes) if layers else None
        rows = {k[5:]: v for k, v in lj.dests.items() if k.startswith("rows.")}
        e = lj.entry
        self.touch(e)
        dt = time.perf_counter() - lj.t0
        self.stats.loaded += 1
        self.stats.loaded_tokens += e.n
        self.stats.load_s += dt
        self.stats.restore_ms.append(dt * 1000.0)
        if len(self.stats.restore_ms) > 4096:
            del self.stats.restore_ms[:2048]
        return obj, rows, meta

    def load(self, key: tuple, dst_for: Callable[[str, list[int], torch.dtype], torch.Tensor | None] = lambda *a: None
             ) -> tuple[Any, dict, dict]:
        """``load_async`` then ``finish_load``: the read on this thread's time (one stream at a time engines)."""

        return self.finish_load(self.load_async(key, dst_for))

    def info(self) -> dict:
        s = self.stats
        with self.lock:
            entries, pending = len(self.index), len(self.pending)
        lat = sorted(s.restore_ms)
        p = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)) if lat else (lambda q: 0.0)
        return {"highwater": self.cfg.highwater, "writers": self.cfg.writers, "direct": self.direct,
                "entries": entries, "pending": pending, "stored_gib": round(self._total() / (1 << 30), 2),
                "stored_bytes": self._total(), "written": s.written, "skipped": s.skipped, "full": s.full,
                "no_space": s.no_space, "deferred": s.deferred, "behind": s.behind,
                "unwritten_evicted": s.unwritten_evicted, "write_failed": s.write_failed, "loaded": s.loaded,
                "load_failed": s.load_failed, "crc_failed": s.crc_failed, "load_unfit": s.load_unfit,
                "pruned": s.pruned,
                "save_s": round(s.save_s, 2),
                "write_s": round(s.write_s, 2), "copy_wait_s": round(s.copy_wait_s, 2), "copy_waits": s.copy_waits,
                "behind_s": round(s.behind_s, 2),
                "written_gib": round(s.written_bytes / (1 << 30), 2), "written_bytes": s.written_bytes,
                "loaded_gib": round(s.loaded_bytes / (1 << 30), 2), "loaded_bytes": s.loaded_bytes,
                "load_s": round(s.load_s, 2), "loaded_tokens": s.loaded_tokens, "restore_ms_p50": p(0.5),
                "restore_ms_p95": p(0.95), "disk_free_bytes": self.disk_free(),
                **{f"unwritten_{k}": v for k, v in sorted(s.unwritten.items())}}
