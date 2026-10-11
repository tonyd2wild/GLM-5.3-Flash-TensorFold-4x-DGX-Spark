"""Models by Hugging Face repo id (resolved in a local cache, no network here) and the families' checks."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tensorfold import families, hub
from tensorfold.cli import _drafter, _generation_config, _model_context, build_parser, main

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def fake_repo(cache: Path, repo_id: str, files: dict[str, str]) -> Path:
    """A repo in Hugging Face's cache layout: refs/main naming a snapshot folder."""

    folder = cache / f"models--{repo_id.replace('/', '--')}"
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(COMMIT)
    snapshot = folder / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    for name, text in files.items():
        (snapshot / name).write_text(text)
    return snapshot


def test_repo_ids_and_local_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert hub.is_repo_id("TensorFold/GLM-5.3-Flash-MLX-4bit-MTP")
    assert not hub.is_repo_id("just-a-name") and not hub.is_repo_id("a/b/c")
    (tmp_path / "local" / "model").mkdir(parents=True)
    assert not hub.is_repo_id("local/model")                 # an existing directory is a directory
    assert hub.resolve("local/model", download=False) == Path("local/model")


def test_cached_repos_resolve_and_missing_ones_say_how_to_pull(tmp_path):
    snapshot = fake_repo(tmp_path, "owner/model", {"config.json": "{}"})
    assert hub.resolve("owner/model", download=False, cache_dir=tmp_path) == snapshot
    assert hub.cached("owner/other", cache_dir=tmp_path) is None
    with pytest.raises(FileNotFoundError, match="tensorfold pull owner/other"):
        hub.resolve("owner/other", download=False, cache_dir=tmp_path)


def test_a_cache_without_refs_still_resolves(tmp_path):
    snapshot = fake_repo(tmp_path, "owner/model", {"config.json": "{}"})
    (tmp_path / "models--owner--model" / "refs" / "main").unlink()
    assert hub.cached("owner/model", cache_dir=tmp_path) == snapshot


def test_a_moved_org_id_finds_the_old_cache_name(tmp_path):
    # the org moved to TensorFold on 2 Oct 2026; a cache pulled before the move kept its Vontra folder name
    snapshot = fake_repo(tmp_path, "Vontra/GLM-5.3-Flash-MLX-4bit-MTP", {"config.json": "{}"})
    assert hub.cached("TensorFold/GLM-5.3-Flash-MLX-4bit-MTP", cache_dir=tmp_path) == snapshot
    assert hub.resolve("TensorFold/GLM-5.3-Flash-MLX-4bit-MTP", download=False, cache_dir=tmp_path) == snapshot


def test_resolve_finishes_a_config_only_cached_model(tmp_path, monkeypatch):
    snapshot = fake_repo(tmp_path, "owner/model", {"config.json": "{}"})
    pulled = []

    def finish(repo_id, *, cache_dir=None):
        pulled.append(repo_id)
        (snapshot / "model.safetensors").write_bytes(b"weights")
        return snapshot

    monkeypatch.setattr(hub, "pull", finish)
    assert hub.resolve("owner/model", cache_dir=tmp_path).resolve() == snapshot.resolve()
    assert pulled == ["owner/model"]


def test_resolve_finishes_missing_shards_but_uses_complete_cache_offline(tmp_path, monkeypatch):
    snapshot = fake_repo(tmp_path, "owner/model", {
        "config.json": "{}",
        "model.safetensors.index.json": json.dumps({"weight_map": {
            "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
    })
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    pulled = []

    def finish(repo_id, *, cache_dir=None):
        pulled.append(repo_id)
        (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
        return snapshot

    monkeypatch.setattr(hub, "pull", finish)
    assert hub.resolve("owner/model", cache_dir=tmp_path).resolve() == snapshot.resolve()
    assert pulled == ["owner/model"]
    assert hub.resolve("owner/model", cache_dir=tmp_path).resolve() == snapshot.resolve()
    assert pulled == ["owner/model"]


def test_resolve_finishes_a_missing_required_mtp_head(tmp_path, monkeypatch):
    snapshot = fake_repo(tmp_path, "owner/model", {
        "config.json": "{}", "model.safetensors": "weights",
    })
    pulled = []

    def finish(repo_id, *, cache_dir=None):
        pulled.append(repo_id)
        (snapshot / "mtp-4bit.safetensors").write_bytes(b"head")
        return snapshot

    monkeypatch.setattr(hub, "pull", finish)
    required = ("mtp-4bit.safetensors",)
    assert not hub._cached_weights_complete(snapshot, required_files=required)
    assert hub.resolve("owner/model", cache_dir=tmp_path, required_files=required) == snapshot
    assert pulled == ["owner/model"]
    assert hub.resolve("owner/model", cache_dir=tmp_path, required_files=required) == snapshot
    assert pulled == ["owner/model"]


def test_resolve_refuses_a_download_that_still_lacks_required_mtp(tmp_path, monkeypatch):
    snapshot = fake_repo(tmp_path, "owner/model", {"config.json": "{}", "model.safetensors": "weights"})
    monkeypatch.setattr(hub, "pull", lambda repo_id, *, cache_dir=None: snapshot)
    with pytest.raises(FileNotFoundError, match="mtp-4bit.safetensors"):
        hub.resolve("owner/model", cache_dir=tmp_path, required_files=("mtp-4bit.safetensors",))


def test_nemotron_pull_checks_its_mtp_head(tmp_path, monkeypatch, capsys):
    repo = "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit"
    snapshot = fake_repo(tmp_path, repo, {
        "config.json": '{"model_type": "nemotron_h", "quantization": {"bits": 4, "group_size": 64}}',
        "model.safetensors": "weights",
    })
    monkeypatch.setattr(hub, "cached", lambda repo_id: snapshot)
    monkeypatch.setattr(hub, "pull", lambda repo_id: snapshot)
    assert main(["pull", repo]) == 1
    assert "missing required files: mtp-4bit.safetensors" in capsys.readouterr().err
    (snapshot / "mtp-4bit.safetensors").write_bytes(b"head")
    assert main(["pull", repo]) == 0
    assert "required model files ready: mtp-4bit.safetensors" in capsys.readouterr().out


def test_native_context_and_sampling_defaults(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h", "text_config": {"max_position_embeddings": 262144}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}))
    assert build_parser().parse_args(["serve", str(tmp_path)]).context is None
    assert _model_context(tmp_path) == 262144
    assert _generation_config(tmp_path) == {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    (tmp_path / "generation_config.json").write_text('{"do_sample": false, "temperature": 1.0}')
    assert _generation_config(tmp_path)["temperature"] == 0.0


def test_context_override_cannot_exceed_model_window(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h", "max_position_embeddings": 4096}))
    monkeypatch.setattr(hub, "resolve", lambda *a, **kw: pytest.fail("should reject before loading weights"))
    assert main(["serve", str(model), "--context", "4097"]) == 1
    assert "exceeds this model's 4096-token window" in capsys.readouterr().err


def test_quantization_is_read_from_the_config():
    assert families.quantization({"quantization": {"bits": 4, "group_size": 32}}) == (4, 32)
    assert families.quantization({"text_config": {"quantization_config": {"bits": 8}}}) == (8, 64)
    assert families.quantization({}) == (None, None)


def write_checkpoint(folder: Path, bits: int, group: int) -> Path:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "nemotron_h", "quantization": {"bits": bits, "group_size": group}}))
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "backbone.embeddings.weight": "model.safetensors"}}))
    return folder


def test_models_lists_the_tested_checkpoints(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    for repo in ("TensorFold/GLM-5.3-Flash-MLX-4bit-MTP", "incoai/GLM-5.3-Flash-DFlash2",
                 "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "mlx-community/gemma-4-26b-a4b-it-4bit"):
        assert repo in out
    for folder in ("glm/flash/v1", "nemotron/lightning/v1", "gemma/v1"):
        assert f"kernels  {folder}" in out


def test_every_family_names_an_importable_kernel_version():
    for family in families.families().values():
        package = family.package
        if not hasattr(package, "load"):
            continue                     # a CUDA-only family (its kernels live in its cuda/ package)
        kernels = importlib.import_module(package.KERNEL_PACKAGE)
        assert kernels.VERSION == package.KERNEL_VERSION == "v1"
        assert family.lanes                 # every family decodes through the lane engine
        names = getattr(package, "MODEL_TYPES", (family.model_type,))     # an alias model_type keeps the package's own
        assert families.kernel_version(family, None).startswith(tuple(f"{name}-v1-" for name in names))


def test_info_reads_a_local_config(tmp_path, capsys):
    folder = write_checkpoint(tmp_path / "nemotron", 4, 64)
    assert main(["info", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "Nemotron 3.5 Lightning" in out
    assert "kernels      nemotron/lightning/v1" in out


def test_auto_drafter_waits_for_a_complete_cached_model(tmp_path, monkeypatch):
    snapshot = fake_repo(tmp_path, "owner/draft", {"config.json": "{}"})
    monkeypatch.setattr(hub, "cached", lambda repo_id: snapshot)
    family = families.families()["glm5_next"]
    assert _drafter(family, "auto") == ""
    (snapshot / "model.safetensors").write_bytes(b"weights")
    assert Path(_drafter(family, "auto")).resolve() == snapshot.resolve()


def test_null_sampling_fields_in_generation_config_keep_the_defaults(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": None, "top_k": 20, "top_p": 0.95, "min_p": None}))
    assert _generation_config(tmp_path) == {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
