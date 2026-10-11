"""NCCL all-gather (and paired send/receive) on the current stream so CUDA graphs capture it; a rank-order sum after
it keeps ranks bit-equal."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

import torch

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4, torch.int8: 0}


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    candidates = [os.environ.get("TF_NCCL_LIB", "")]
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates += glob.glob("/usr/lib/*/libnccl.so.2") + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2")
    candidates += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl*.so*"))
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("libnccl not found (set TF_NCCL_LIB)")


class NCCL:
    def __init__(self, rank: int, world: int, master: str, port: int) -> None:
        from datetime import timedelta

        from torch.distributed import TCPStore

        self.rank, self.world = rank, world
        # TF_NCCL_RING=1: the ranks are cabled as a ring in rank order, each only to the next and the previous, so
        # exchange_all passes every message along the ring over two-rank links with the neighbours (_exchange_ring)
        self.ring = os.environ.get("TF_NCCL_RING", "0").strip() == "1" and world > 2
        # TF_NCCL_RING_INPLACE: 1 (the default) runs _exchange_ring's link all-gathers in place, each link's steps on
        # its own stream (_ring_links); 0 keeps patch 0098's steps (_ring_steps). The same bytes arrive either way
        self.inplace = os.environ.get("TF_NCCL_RING_INPLACE", "1").strip() != "0"
        self.links: dict[int, ctypes.c_void_p] = {}
        self.link_streams: dict[int, torch.cuda.Stream] = {}
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        for name in ("ncclSend", "ncclRecv"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p]
        lib.ncclGroupStart.argtypes = []
        lib.ncclGroupEnd.argtypes = []
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_nccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_nccl_uid")
            ctypes.memmove(ctypes.addressof(uid), raw, 128)
        self.comm = ctypes.c_void_p()
        torch.cuda.current_device()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank))
        if self.ring:
            self._open_links()

    def _open_links(self) -> None:
        """TF_NCCL_RING: a two-rank communicator with each neighbour (link a joins rank a, its rank 0, and rank a + 1,
        its rank 1), opened in link order, so no rank waits on one that waits on it. Each reads its own NCCL graph
        file (TF_NCCL_RING_GRAPHS: the one toward the previous rank, then the one toward the next), so its channels
        run over the cable to that neighbour, a port and its PCIe twin. Point-to-point sends take no graph file: NCCL
        receives them on a device of its choosing, and on a ring cabled port to like port that is half the time a
        port without the peer."""

        N, me = self.world, self.rank
        files = (os.environ.get("TF_NCCL_RING_GRAPHS", "") + ",").split(",")[:2]
        self.lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p]
        saved = os.environ.get("NCCL_GRAPH_FILE")
        try:
            for a in sorted({(me - 1) % N, me}):
                first = a == me                         # this rank is the link's rank 0: the link to the next rank
                uid = _UniqueId()
                if first:
                    self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
                    self.store.set(f"tf_nccl_link_{a}", bytes(uid.internal))
                else:
                    ctypes.memmove(ctypes.addressof(uid), self.store.get(f"tf_nccl_link_{a}"), 128)
                graph = files[1] if first else files[0]
                if graph:
                    os.environ["NCCL_GRAPH_FILE"] = graph
                else:
                    os.environ.pop("NCCL_GRAPH_FILE", None)
                comm = ctypes.c_void_p()
                self._check(self.lib.ncclCommInitRank(ctypes.byref(comm), 2, uid, 0 if first else 1))
                self.links[1 if first else -1] = comm
                self.link_streams[1 if first else -1] = torch.cuda.Stream()
        finally:
            if saved is None:
                os.environ.pop("NCCL_GRAPH_FILE", None)
            else:
                os.environ["NCCL_GRAPH_FILE"] = saved

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order (contiguous tensors, same dtype)."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllGather(send.data_ptr(), recv.data_ptr(), send.numel(), _DTYPES[send.dtype],
                                           self.comm, stream))

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None:
        """Every rank finishes ``label`` before any goes on; a rank missing after ``timeout`` s is named."""

        import time
        from datetime import timedelta

        self.store.set(f"tf_ready/{label}/{self.rank}", "1")
        others = [r for r in range(self.world) if r != self.rank]
        started = time.monotonic()
        while True:
            try:
                self.store.wait([f"tf_ready/{label}/{r}" for r in others], timedelta(seconds=every))
                return
            except Exception as exc:                  # noqa: BLE001  (the store's timeout; anything else goes up)
                if "timeout" not in str(exc).lower():
                    raise
            waited = time.monotonic() - started
            missing = ", ".join(str(r) for r in others)
            if waited >= timeout:
                raise RuntimeError(f"rank {self.rank} finished {label} but rank {missing} has not after "
                                   f"{waited / 60:.0f} min: check that rank's log (a CUDA extension build waiting on "
                                   "a lock names the lock there)")
            print(f"[tensorfold] rank {self.rank} finished {label}; waiting for rank {missing} ({waited:.0f} s)",
                  flush=True)

    def exchange(self, sends: list[torch.Tensor], recvs: list[torch.Tensor], peer: int) -> None:
        """One NCCL group on the current stream: every ``sends[i]`` to ``peer`` and ``recvs[i]`` from it (contiguous
        tensors; the peer's matching call has the same count, sizes and dtypes in the same order)."""

        if len(sends) != len(recvs) or peer == self.rank or not 0 <= peer < self.world:
            raise ValueError("exchange: one receive per send, from another rank")
        for s, r in zip(sends, recvs):
            if s.numel() != r.numel() or s.dtype != r.dtype or not (s.is_contiguous() and r.is_contiguous()):
                raise ValueError("exchange: each send and receive must be contiguous and alike in size and dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclGroupStart())
        try:
            for s, r in zip(sends, recvs):
                if s.numel():
                    self._check(self.lib.ncclSend(s.data_ptr(), s.numel(), _DTYPES[s.dtype], peer, self.comm, stream))
                    self._check(self.lib.ncclRecv(r.data_ptr(), r.numel(), _DTYPES[r.dtype], peer, self.comm, stream))
        finally:
            self._check(self.lib.ncclGroupEnd())

    def exchange_all(self, sends: dict[int, list[torch.Tensor]], recvs: dict[int, list[torch.Tensor]]) -> None:
        """One NCCL group on the current stream with several peers: every ``sends[p][i]`` to peer p and
        ``recvs[p][i]`` from it (an all-to-all of row blocks; each peer's matching call names this rank with the
        same sizes and dtypes in the same order)."""

        if set(sends) != set(recvs) or any(p == self.rank or not 0 <= p < self.world for p in sends):
            raise ValueError("exchange_all: sends and receives with the same other ranks")
        for p in sends:
            if len(sends[p]) != len(recvs[p]):
                raise ValueError("exchange_all: one receive per send")
            for s, r in zip(sends[p], recvs[p]):
                if s.numel() != r.numel() or s.dtype != r.dtype or not (s.is_contiguous() and r.is_contiguous()):
                    raise ValueError("exchange_all: each send and receive must be contiguous and alike")
        if self.ring:
            self._exchange_ring(sends, recvs)
            return
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclGroupStart())
        try:
            for p in sorted(sends):
                for s, r in zip(sends[p], recvs[p]):
                    if s.numel():
                        self._check(self.lib.ncclSend(s.data_ptr(), s.numel(), _DTYPES[s.dtype], p, self.comm, stream))
                        self._check(self.lib.ncclRecv(r.data_ptr(), r.numel(), _DTYPES[r.dtype], p, self.comm, stream))
        finally:
            self._check(self.lib.ncclGroupEnd())

    def _exchange_ring(self, sends: dict[int, list[torch.Tensor]], recvs: dict[int, list[torch.Tensor]]) -> None:
        """exchange_all on a ring where each rank reaches only its neighbours (TF_NCCL_RING=1): a message from rank s
        to rank t goes the shorter way round, one hop a step, a message to the rank opposite (N even) half each way.
        Each of the floor(N / 2) steps is a two-rank all-gather on each link (_open_links), each on the link's own
        stream (two communicators' kernels on one stream run one after the other: measured on four Sparks, both
        links 10 GB/s each way on one stream, 14 on two): a rank packs what it sends that neighbour, as bytes, and
        gets back what the neighbour sends it. Every rank must name all the others, and the i-th tensor must have the same size
        and dtype for every peer (row blocks of equal shares), so that a relaying rank knows each message's size.
        recvs end up with exactly exchange_all's bytes.

        TF_NCCL_RING_INPLACE=1 (the default, ``_ring_links``): each link's all-gather runs in place and each link's
        steps run on its own stream, joined to the other link's only where bytes pass from one to the other.
        TF_NCCL_RING_INPLACE=0: patch 0098's ``_ring_steps``."""

        N, me = self.world, self.rank
        peers = [p for p in range(N) if p != me]
        if sorted(sends) != peers:
            raise ValueError("exchange_all (TF_NCCL_RING): every other rank, in one call")
        k = len(sends[peers[0]])
        spec = [(sends[peers[0]][i].numel(), sends[peers[0]][i].dtype) for i in range(k)]
        for p in peers:
            if len(sends[p]) != k or any((t.numel(), t.dtype) != spec[i] for d in (sends[p], recvs[p])
                                         for i, t in enumerate(d)):
                raise ValueError("exchange_all (TF_NCCL_RING): the same sizes and dtypes for every peer")

        def segments(s: int, t: int, i: int) -> list[tuple[int, int, int, int]]:
            """(lo, hi, direction, hops) of the message s -> t, tensor i, as flat element ranges."""
            n, d = spec[i][0], (t - s) % N
            if 2 * d < N:
                return [(0, n, 1, d)]
            if 2 * d > N:
                return [(0, n, -1, N - d)]
            return [(0, n // 2, 1, d), (n // 2, n, -1, d)]

        def moving(src: int, direction: int, h: int) -> list[tuple[int, int, int, int, int]]:
            """The messages from src that travel ``direction`` on their h-th hop: (t, i, lo, hi, hops), in the one
            order both ends of a link list them."""
            return [(t, i, lo, hi, hops) for t in range(N) if t != src for i in range(k)
                    for lo, hi, dr, hops in segments(src, t, i) if dr == direction and hops >= h and hi > lo]

        if self.inplace:
            self._ring_links(sends, recvs, spec, moving)
        else:
            self._ring_steps(sends, recvs, spec, moving)

    def _ring_links(self, sends, recvs, spec, moving) -> None:
        """_exchange_ring with fewer copies and joins (TF_NCCL_RING_INPLACE=1). Each step's link all-gather runs in
        place: a rank packs what it sends straight into its own slot of the link's output (its slot is its rank on
        the link: 0 on the link to the next rank, 1 on the one to the previous), so NCCL copies nothing locally, and
        nothing is zero-filled (the two ends of a link name the same count: what one sends the other expects). Each
        link's whole chain (pack, all-gather, the unpacking of its last-hop bytes) runs on the link's stream: step 1
        after the caller's stream, a later step on link d after the other link's previous step (the bytes it relays
        arrived there; an event, so not after that link's later steps) and after its own; the caller's stream waits
        for both links at the end. Patch 0098 packed and unpacked on the caller's stream between steps, both links
        joined there. Launch order on every rank is _ring_steps': the link to the previous rank first (rank 0: link 0
        first), so that link kernels that cannot run at once never wait on each other round the ring."""

        N, me = self.world, self.rank
        main = torch.cuda.current_stream()
        held: dict[tuple[int, int, int, int], torch.Tensor] = {}   # relayed bytes, by (s, t, i, lo)
        order = sorted((1, -1), key=lambda d: (me - 1) % N if d == -1 else me)
        done: dict[int, torch.cuda.Event] = {}         # each link's last step so far
        used = set()
        for h in range(1, N // 2 + 1):
            step_done = {}
            for direction in order:
                src_out = (me - direction * (h - 1)) % N    # what leaves on this link: travelling `direction`
                src_in = (me + direction * h) % N           # what arrives on it: travelling the other way
                leave = moving(src_out, direction, h)
                arrive = moving(src_in, -direction, h)
                nbytes = lambda segs: sum((hi - lo) * spec[i][1].itemsize for t, i, lo, hi, hops in segs)
                n_out = nbytes(leave)
                L = max(n_out, nbytes(arrive))
                if L == 0:
                    continue
                side = self.link_streams[direction]
                if h == 1:
                    side.wait_stream(main)
                elif -direction in done:                    # the bytes it relays came in on the other link
                    side.wait_event(done[-direction])
                used.add(direction)
                with torch.cuda.stream(side):
                    out = []
                    for t, i, lo, hi, hops in leave:
                        if h == 1:
                            out.append(sends[t][i].view(-1)[lo:hi].view(torch.int8))
                        else:
                            chunk = held.pop((src_out, t, i, lo))
                            chunk.record_stream(side)           # the other link's buffer, read here
                            out.append(chunk)
                    both = torch.empty(2 * L, dtype=torch.int8, device="cuda")
                    mine = both[:L] if direction == 1 else both[L:]
                    if len(out) == 1:
                        mine[:n_out].copy_(out[0])
                    elif out:
                        torch.cat(out, out=mine[:n_out])
                    self._check(self.lib.ncclAllGather(mine.data_ptr(), both.data_ptr(), L, _DTYPES[torch.int8],
                                                       self.links[direction], side.cuda_stream))
                    theirs = both[L:] if direction == 1 else both[:L]   # the neighbour's slot
                    at = 0
                    for t, i, lo, hi, hops in arrive:
                        n = (hi - lo) * spec[i][1].itemsize
                        chunk = theirs[at:at + n]
                        at += n
                        if hops == h:                       # its last hop: this rank is t
                            recvs[src_in][i].view(-1)[lo:hi].view(torch.int8).copy_(chunk)
                        else:
                            held[(src_in, t, i, lo)] = chunk
                    ev = torch.cuda.Event()
                    ev.record(side)
                    step_done[direction] = ev
            done.update(step_done)
        if held:
            raise RuntimeError("exchange_all (TF_NCCL_RING): relayed bytes left over")
        # the caller's stream after both links (the sends and receives are the caller's: anything after this call on
        # its stream, the reuse of their memory included, comes after the links' reads and writes)
        for direction in order:
            if direction in used:
                main.wait_stream(self.link_streams[direction])

    def _ring_steps(self, sends, recvs, spec, moving) -> None:
        """Patch 0098's _exchange_ring (TF_NCCL_RING_INPLACE=0): each step packs on the caller's stream, runs both
        links' all-gathers (a separate zero-filled input each), then unpacks on the caller's stream."""

        N, me = self.world, self.rank
        held: dict[tuple[int, int, int, int], torch.Tensor] = {}   # bytes this rank relays, by (s, t, i, lo)
        main = torch.cuda.current_stream()
        for h in range(1, N // 2 + 1):
            work = []
            for direction in (1, -1):                   # the link to the next rank, then the one to the previous
                src_out = (me - direction * (h - 1)) % N    # what leaves on this link: travelling `direction`
                src_in = (me + direction * h) % N           # what arrives on it: travelling the other way
                out = [sends[t][i].view(-1)[lo:hi].view(torch.int8) if h == 1 else held.pop((src_out, t, i, lo))
                       for t, i, lo, hi, hops in moving(src_out, direction, h)]
                arrive = moving(src_in, -direction, h)
                size = lambda segs: sum((hi - lo) * spec[i][1].itemsize for t, i, lo, hi, hops in segs)
                L = max(sum(x.numel() for x in out), size(arrive))
                if L == 0:
                    continue
                buf = torch.zeros(L, dtype=torch.int8, device="cuda")
                if out:
                    torch.cat(out, out=buf[:sum(x.numel() for x in out)])
                both = torch.empty(2 * L, dtype=torch.int8, device="cuda")
                work.append((direction, buf, both, src_in, arrive))
            # every rank launches its links in link order (the one to the previous rank first, except on rank 0,
            # whose link 0 comes before link N - 1): without an NCCL group around two communicators, kernels that
            # run one at a time would otherwise wait on each other round the ring
            work.sort(key=lambda w: (me - 1) % N if w[0] == -1 else me)
            for direction, buf, both, _, _ in work:
                side = self.link_streams[direction]
                side.wait_stream(main)
                buf.record_stream(side)
                both.record_stream(side)
                self._check(self.lib.ncclAllGather(buf.data_ptr(), both.data_ptr(), buf.numel(), _DTYPES[torch.int8],
                                                   self.links[direction], side.cuda_stream))
            for direction, *_ in work:
                main.wait_stream(self.link_streams[direction])
            for direction, buf, both, src_in, arrive in work:
                theirs = both[buf.numel():] if direction == 1 else both[:buf.numel()]   # the neighbour's slot
                at = 0
                for t, i, lo, hi, hops in arrive:
                    nbytes = (hi - lo) * spec[i][1].itemsize
                    chunk = theirs[at:at + nbytes]
                    at += nbytes
                    if hops == h:                       # its last hop: this rank is t
                        recvs[src_in][i].view(-1)[lo:hi].view(torch.int8).copy_(chunk)
                    else:
                        held[(src_in, t, i, lo)] = chunk

    def barrier(self) -> None:
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()
