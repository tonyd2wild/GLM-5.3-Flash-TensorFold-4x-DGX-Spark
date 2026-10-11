"""Packed CUDA format contracts and kernel address arithmetic without GPU runtimes."""

import ast
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.cuda.kernels.affine import BITS, GROUPS, input_slice, packed_shape

ROOT = Path(__file__).parents[1] / "src/tensorfold"


class Array(np.ndarray):
    def to(self, dtype):
        return np.asarray(self).astype(dtype).view(Array)


def array(value):
    return np.asarray(value).view(Array)


class Pointer:
    def __init__(self, values, offsets=0):
        self.values, self.offsets = np.asarray(values).reshape(-1), offsets

    def __add__(self, offsets):
        return Pointer(self.values, self.offsets + offsets)


class Language:
    uint32, float32, bfloat16 = np.uint32, np.float32, np.float32
    program = (0, 0)

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < len(pointer.values)))
        result = np.full(offsets.shape, other, dtype=pointer.values.dtype)
        result[mask] = pointer.values[offsets[mask]]
        return array(result)

    @staticmethod
    def store(pointer, value, mask=True):
        offsets, value, mask = np.broadcast_arrays(pointer.offsets, value, mask)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < len(pointer.values)))
        pointer.values[offsets[mask]] = value[mask]

    @staticmethod
    def where(condition, a, b):
        return array(np.where(condition, a, b))

    @classmethod
    def program_id(cls, axis):
        return cls.program[axis]

    arange = staticmethod(lambda a, b: array(np.arange(a, b)))
    zeros = staticmethod(lambda shape, dtype: array(np.zeros(shape, dtype=dtype)))
    trans = staticmethod(lambda a: a.T)
    sum = staticmethod(lambda a, axis: array(np.sum(a, axis=axis)))
    dot = staticmethod(lambda a, b, **kw: array(np.sum(a[:, :, None] * b[None, :, :], axis=1)))


def kernels():
    tree = ast.parse((ROOT / "cuda/kernels/affine_kernels.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("codes", "matmul")]
    for function in functions:
        function.decorator_list = []
        for argument in function.args.args:
            argument.annotation = None
    namespace = {"tl": Language}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "affine_kernel_cpu_contract", "exec"), namespace)
    return namespace


def pack(values, bits):
    result = []
    for row in values:
        stream = sum(int(value) << (i * bits) for i, value in enumerate(row))
        result.append([(stream >> shift) & 0xFFFFFFFF for shift in range(0, len(row) * bits, 32)])
    return np.array(result, dtype=np.uint32)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("group", GROUPS)
def test_every_packed_format_crosses_words_and_rank_boundaries_exactly(bits, group):
    rng = np.random.default_rng(811)
    values = rng.integers(0, 1 << bits, size=(5, group * 4), dtype=np.uint32)
    words = pack(values, bits)
    assert packed_shape(words.shape, (5, 4), (5, 4), bits, group) == values.shape
    decode = kernels()["codes"]
    rows, columns = array(np.arange(8)[:, None]), array(np.arange(group * 4)[None, :])
    decoded = decode(Pointer(words), rows, columns, rows < 5, words.shape[1], bits)
    np.testing.assert_array_equal(decoded[:5], values)
    assert not decoded[5:].any()
    for rank in (0, 1):
        (a, b), (g0, g1) = input_slice(group * 4, bits, group, rank)
        part = words[:, a:b].copy()
        got = decode(Pointer(part), rows, array(np.arange(group * 2)[None, :]), rows < 5, b - a, bits)
        np.testing.assert_array_equal(got[:5], values[:, rank * group * 2:(rank + 1) * group * 2])
        assert (g0, g1) == (rank * 2, rank * 2 + 2)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("group", GROUPS)
def test_kernel_rows_match_solo_with_masked_output_and_partial_row_tiles(bits, group):
    rng = np.random.default_rng(93)
    k, n, m = group * 2, 35, 19
    values = rng.integers(0, 1 << bits, size=(n, k), dtype=np.uint32)
    packed = pack(values, bits)
    scales = np.full((n, 2), 0.25, dtype=np.float32)
    biases = np.full((n, 2), -0.5, dtype=np.float32)
    x = rng.integers(-2, 3, size=(m, k)).astype(np.float32) * 0.5
    kernel = kernels()["matmul"]
    def run(inputs):
        out = np.empty((len(inputs), n), dtype=np.float32)
        for row in range((len(inputs) + 15) // 16):
            for column in range((n + 31) // 32):
                Language.program = row, column
                kernel(Pointer(inputs), Pointer(packed), Pointer(scales), Pointer(biases), Pointer(out),
                       len(inputs), n, k, bits, group)
        return out
    actual = run(x)
    expected = np.sum(x[:, None, :] * (values.astype(np.float32) * 0.25 - 0.5)[None, :, :], axis=2)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[17:18], run(x[17:18]))


@pytest.mark.parametrize("bits,group,shape", [(1, 64, (4, 16)), (8, 16, (4, 64)), (3, 32, (4, 4))])
def test_bad_layout_is_rejected_before_dispatch(bits, group, shape):
    with pytest.raises(ValueError):
        packed_shape(shape, (4, 2), (4, 2), bits, group)


def test_parallel_split_refuses_partial_groups():
    with pytest.raises(ValueError):
        input_slice(384, 5, 128, 0)


def test_qlinear_preserves_packed_width_and_scale_precision():
    tree = ast.parse((ROOT / "cuda/mlx4/linear.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QLinear")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"dataclass": dataclass, "cached_property": cached_property, "torch": SimpleNamespace(bfloat16="bf16")}
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, "affine_weight_metadata", "exec"), namespace)
    QLinear = namespace["QLinear"]
    metadata = SimpleNamespace(shape=(17, 4), dtype="fp16")
    weight = SimpleNamespace(shape=(17, 24))
    q = QLinear(weight, metadata, metadata, bits=3, gs=64)
    assert (q.n, q.k, q.fast) == (17, 256, False)
    assert not QLinear(weight, metadata, metadata, bits=4, gs=64).fast
    metadata.dtype = "bf16"
    assert QLinear(weight, metadata, metadata, bits=4, gs=64).fast
