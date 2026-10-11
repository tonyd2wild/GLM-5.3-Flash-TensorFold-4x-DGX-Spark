"""`python tests/limit_scenarios.py SCENARIO LIMIT` (0: none): one scenario's output digests as JSON, fresh process."""

from __future__ import annotations

import hashlib
import json
import sys

import thread_limit

thread_limit.install(int(sys.argv[2]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

out: dict[str, str] = {}


def keep(name: str, *arrays: mx.array) -> None:
    h = hashlib.sha256()
    for a in arrays:
        a = mx.contiguous(a)
        mx.eval(a)
        h.update(np.asarray(a.reshape(-1).view(mx.uint8)).tobytes())
    out[name] = h.hexdigest()[:16]


def _weights(n: int, k: int, seed: int) -> tuple[mx.array, ...]:
    w = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=4)


def simd_qmm() -> None:
    from tensorfold.kernels.qwen.dense.v1 import row_matmul
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm as sq

    back = row_matmul.simd_qmm_backend()
    for n, k in ((48, 5120), (1024, 5120), (5120, 6144), (6144, 5120), (2048, 1024)):
        q, s, b = _weights(n, k, 7)
        out[f"check {n}x{k}"] = str(sq.check(q, s, b))
        x = (mx.random.normal((64, k), key=mx.random.key(3)) * 0.5).astype(mx.bfloat16)
        for rows in (1, 2, 4, 8, 9, 16, 17, 24, 33, 64):
            keep(f"qmm {n}x{k} {rows}", sq.qmm(x[:rows], q, s, b))
            if rows <= 16:
                keep(f"backend {n}x{k} {rows}", back.qmm(x[:rows], q, s, b, 64))


def sampling() -> None:
    from types import SimpleNamespace

    from tensorfold.engine import gpu_sampling, topk

    for vocab in (248320, 131072):
        logits = (mx.random.normal((4, vocab), key=mx.random.key(vocab)) * 3).astype(mx.bfloat16)
        ids = mx.arange(vocab).astype(mx.uint32)
        for temp, top_p, top_k in ((1.0, 0.95, 20), (0.7, 0.9, 0), (1.0, 1.0, 0), (1.3, 0.5, 64)):
            rows = [SimpleNamespace(temperature=temp, top_p=top_p, top_k=top_k, seed=11 * r + 1) for r in range(4)]
            keep(f"sample {vocab} {temp} {top_p} {top_k}", gpu_sampling.sample_rows(logits, rows, [5, 6, 7, 900]))
            keep(f"sample ids {vocab} {temp} {top_p} {top_k}",
                 gpu_sampling.sample_rows(logits, rows, [1, 2, 3, 4], ids))
        for k in (1, 8, 64):
            keep(f"topk {vocab} {k}", *topk.topk_rows(logits, k))


def nemotron() -> None:
    from tensorfold.kernels.nemotron.lightning.v1 import kernels as nk, rows

    dims = 2688
    h = mx.random.normal((16, dims), key=mx.random.key(1)).astype(mx.bfloat16)
    d = mx.random.normal((16, dims), key=mx.random.key(2)).astype(mx.bfloat16)
    w = mx.random.uniform(0.5, 1.5, (dims,), key=mx.random.key(3)).astype(mx.bfloat16)
    eps = mx.array([1e-5], dtype=mx.float32)
    routed = mx.random.normal((16, 6, dims), key=mx.random.key(4)).astype(mx.bfloat16)
    weights = mx.random.uniform(0, 1, (16, 6), key=mx.random.key(5))
    for r in (1, 5, 16):
        keep(f"add_norm {r}", *nk.add_norm(h[:r], d[:r], w, eps))
        keep(f"add_norm_moe {r}", *nk.add_norm_moe(h[:r], routed[:r], weights[:r], d[:r], w, eps))
    for n, k in ((4096, 2688), (2688, 4096), (3712, 2688)):
        q, sc, b = _weights(n, k, 9)
        x = (mx.random.normal((16, k), key=mx.random.key(6)) * 0.5).astype(mx.bfloat16)
        for r in range(1, 17):
            keep(f"qmv {n}x{k} {r}", rows.qmv(x[:r], q, sc, b, 64))


def gemma() -> None:
    from gemma4_tiny import tiny_text, tokens

    from tensorfold.families.gemma4.model import Gemma4
    from tensorfold.kernels.gemma.v1 import glue
    from tensorfold.kernels.inputs import ints

    eps = mx.array([1e-6], dtype=mx.float32)
    dims, heads, kv = 2816, 4, 2
    for head_dim in (256, 512):                     # the checkpoint's sliding and full layers
        for values_are_keys in (False, True):
            width = (heads + kv * (1 if values_are_keys else 2)) * head_dim
            q, sc, b = _weights(width, dims, 11)
            x = (mx.random.normal((3, dims), key=mx.random.key(12)) * 0.5).astype(mx.bfloat16)
            qw = mx.random.uniform(0.5, 1.5, (head_dim,), key=mx.random.key(13)).astype(mx.bfloat16)
            inv = mx.array(np.linspace(1.0, 1e-4, head_dim // 2, dtype=np.float32))
            keep(f"qkv_rows {head_dim} {values_are_keys}",
                 *glue.qkv_rows(x, q, sc, b, 64, qw, qw, inv, ints([7, 900, 40000]), eps, heads=heads,
                                kv_heads=kv, head_dim=head_dim, values_are_keys=values_are_keys))
    model = Gemma4(tiny_text(), backend="rows", check=False)
    cache = model.make_cache()
    keep("prefill", model.prefill(mx.array([tokens(20)], dtype=mx.uint32), cache))
    for i, token in enumerate(tokens(4, seed=5)):
        keep(f"step {i}", model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache)))
    keep("window", model.head(model.hidden(mx.array([tokens(6, seed=7)], dtype=mx.uint32), cache)))


SCENARIOS = {"simd_qmm": simd_qmm, "sampling": sampling, "nemotron": nemotron, "gemma": gemma}


if __name__ == "__main__":
    from tensorfold.kernels import threads

    SCENARIOS[sys.argv[1]]()
    print(json.dumps({"out": out, "largest": thread_limit.largest, "reserved": thread_limit.reserved,
                      "guessed": len(threads.guessed),
                      "fitted": {repr(k): v for k, v in threads._fitted.items()}}))
