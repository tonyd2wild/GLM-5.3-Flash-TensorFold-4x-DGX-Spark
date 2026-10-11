"""Family metadata gates and CLI routing are exercised without loading accelerator frameworks."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from tensorfold import cli, families
from tensorfold.families import nemotron_h


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def configuration(bits=4, group=64, **overrides):
    return {"quantization": {"bits": bits, "group_size": group, **overrides}}


def test_family_metadata_prioritizes_modern_quantization_and_skips_empty_overrides():
    value = configuration(8, 128, **{"model.layers.0.q_proj": {}, "model.layers.0.k_proj": False,
                                    "model.layers.0.v_proj": {"bits": 3}})
    value["quantization_config"] = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    assert families.quant_method(value) == "mlx" and families.quantization(value) == (8, 128)
    assert families.layer_quantization(value) == {"model.layers.0.v_proj": (3, 64, "affine")}
    assert "8-bit" in families.describe_quantization(value)


@pytest.mark.parametrize("package,group", [(nemotron_h, 64)])
@pytest.mark.parametrize("bits", [2, 3, 5, 6, 8])
def test_moe_families_still_refuse_other_bit_widths_on_cuda(package, group, bits):
    family = families.Family(package.MODEL_TYPES[0], package.TITLE, package.__name__, True)
    with pytest.raises(ValueError, match="4-bit"):
        families.require_readable(family, configuration(bits, group), "cuda")


@pytest.mark.parametrize("bits", [2, 3, 5, 6, 8])
def test_moe_family_metal_preflight_still_refuses_other_widths(tmp_path, bits):
    (tmp_path / "config.json").write_text(json.dumps(configuration(bits, 64)))
    with pytest.raises(ValueError, match="4-bit"):
        nemotron_h.check(tmp_path)


@pytest.mark.parametrize("lane_kernels", ["auto", "on", "off"])
def test_cli_passes_kernel_choice_to_family_loader_without_allocating(tmp_path, monkeypatch, lane_kernels):
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--lane-kernels", lane_kernels,
                                          "--no-drafts", "--no-update-check"])
    core, mlx = ModuleType("mlx.core"), ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    calls = []

    class AtLoader(Exception):
        pass

    def load(path, **options):
        calls.append((path, options))
        raise AtLoader

    family = NS(package=NS(load=load), title="A dense family", model_type="test")
    with pytest.raises(AtLoader):
        cli._serve_mlx(args, family, tmp_path, 4096, (), 1024**3)
    assert calls[0][0] == tmp_path and calls[0][1]["lane_kernels"] == lane_kernels
    assert calls[0][1]["drafter"] == ""
