"""Snapshot identities include prompt kernels, their modes, and library versions."""

from importlib import metadata
from types import SimpleNamespace

from tensorfold import cli, families
from tensorfold.server import app


def test_snapshot_identity_changes_with_mlx_lm_version(monkeypatch, tmp_path):
    captured = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(kwargs["model_id"])
        raise Captured

    monkeypatch.setattr(app, "ChatApp", capture)
    monkeypatch.setattr(families, "kernel_version", lambda *args: "fixed-kernels")
    family = SimpleNamespace(title="fake", model_type="fake", package=SimpleNamespace(load=lambda *a, **k: (None, None)))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--prompt-cache-gib", "0"])
    for version in ("0.31.3", "0.31.4"):
        monkeypatch.setattr(metadata, "version", lambda name, value=version: value)
        try:
            cli._serve_mlx(args, family, tmp_path, 0, [], 1 << 30)
        except Captured:
            pass
    assert len(captured) == 2 and captured[0] != captured[1]
    assert "mlx_lm=0.31.3" in captured[0] and "mlx_lm=0.31.4" in captured[1]
