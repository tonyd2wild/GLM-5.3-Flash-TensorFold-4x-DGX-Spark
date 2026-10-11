"""The CLI's CUDA path: backend choice and argument checks that run before any GPU work (any machine)."""

import argparse
from types import SimpleNamespace

import pytest

from tensorfold import cli


def _family(**members):
    return SimpleNamespace(title="Test family", package=SimpleNamespace(**members))


def test_auto_backend_follows_the_platform(monkeypatch):
    both = _family(load=lambda *a, **k: None, cuda_engine=lambda *a, **k: None)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    assert cli._backend("auto", both) == "mlx"
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli._backend("auto", both) == "cuda"


def test_a_family_serves_only_the_backends_it_has():
    with pytest.raises(ValueError, match="no CUDA engine"):
        cli._backend("cuda", _family(load=lambda *a, **k: None))
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        cli._backend("mlx", _family(cuda_engine=lambda *a, **k: None))


def test_two_gpus_need_a_master_before_anything_loads(tmp_path):
    called = []
    family = _family(cuda_engine=lambda *a, **k: called.append(k))
    args = argparse.Namespace(tp=2, rank=0, master="", master_port=29551, no_drafts=True, drafter="none",
                              mtp_drafts=None, name="", model=str(tmp_path))
    with pytest.raises(ValueError, match="--master"):
        cli._serve_cuda(args, family, tmp_path)
    args.tp, args.rank = 1, 1
    with pytest.raises(ValueError, match="--rank 1 needs --tp 2"):
        cli._serve_cuda(args, family, tmp_path)
    assert not called


def test_four_gpus_only_for_a_family_that_splits_over_four(tmp_path):
    called = []
    family = _family(cuda_engine=lambda *a, **k: called.append(k))
    args = argparse.Namespace(tp=4, rank=0, master="192.0.2.11", master_port=29551, no_drafts=True, drafter="none",
                              mtp_drafts=None, name="", model=str(tmp_path))
    with pytest.raises(ValueError, match="--tp 4: Test family runs on one or two GPUs"):
        cli._serve_cuda(args, family, tmp_path)
    args.tp, args.rank = 2, 3
    with pytest.raises(ValueError, match="--rank 3 needs --tp 4"):
        cli._serve_cuda(args, family, tmp_path)
    assert not called

def test_serve_parses_the_cuda_flags():
    args = cli.build_parser().parse_args(["serve", "owner/model", "--tp", "2", "--rank", "1", "--master", "192.0.2.11"])
    assert (args.backend, args.tp, args.rank, args.master, args.master_port) == ("auto", 2, 1, "192.0.2.11", 29551)


@pytest.mark.parametrize("override, expected", [(None, 128), (0, 0), (64, 64)])
def test_cuda_dispatch_keeps_the_resolved_context(tmp_path, monkeypatch, override, expected):
    import json

    from tensorfold import families, hub

    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 128}))
    family = _family(cuda_engine=lambda *a, **k: None)
    family.model_type = "test"
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: tmp_path)
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, found, path, context: seen.append(context) or 0)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-update-check"]
    if override is not None:
        command += ["--context", str(override)]
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert seen == [expected]


def test_a_cuda_start_dumps_its_stacks_on_sigusr1(tmp_path, monkeypatch):
    """`kill -USR1 <pid>` shows where a CUDA start waits, as on the Mac: the handler is in place before any load."""

    import json
    import signal

    from tensorfold import families, hub

    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 128}))
    family = _family(cuda_engine=lambda *a, **k: None)
    family.model_type = "test"
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: tmp_path)
    events = []
    monkeypatch.setattr("faulthandler.register", lambda signum, **k: events.append(("register", signum, k)))
    monkeypatch.setattr(cli, "_serve_cuda", lambda *a: events.append(("serve",)) or 0)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-update-check"]
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert events == [("register", signal.SIGUSR1, {"all_threads": True}), ("serve",)]


def test_cuda_admission_metadata_does_not_enlarge_the_engine_cache(tmp_path, monkeypatch, capsys):
    import tensorfold.cuda.server as server

    made, served = [], []
    engine = SimpleNamespace(max_len=8192)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: served.append(k) or
                        SimpleNamespace(effective_context_window=8185))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    assert cli._serve_cuda(args, family, tmp_path, 262144) == 0
    assert made[0]["context"] == 262144
    assert made[0]["context_explicit"] is False
    assert served[0]["context_window"] == 262144
    assert "context: 8185" in capsys.readouterr().out


def test_the_prompt_precision_flag_parses_and_help_states_the_default():
    from tensorfold.cuda import prompt_precision

    parser = cli.build_parser()
    assert getattr(parser.parse_args(["serve", "owner/model"]), "prefill_fp8", None) is None   # the default applies
    assert parser.parse_args(["serve", "owner/model", "--prefill-fp8"]).prefill_fp8 is True
    assert parser.parse_args(["serve", "owner/model", "--no-prefill-fp8"]).prefill_fp8 is False
    default = "FP8" if prompt_precision.FP8_BY_DEFAULT else "bf16"         # one constant picks the default
    serve = next(a for a in parser._actions if a.dest == "command").choices["serve"]
    flag = next(a for a in serve._actions if a.dest == "prefill_fp8")
    assert "e4m3" in flag.help and f"Default: {default} activations" in flag.help   # --help states both precisions


@pytest.mark.parametrize("fast,flags,fp8", [(True, [], None), (True, ["--prefill-fp8"], True),
                                            (True, ["--no-prefill-fp8"], False), (False, ["--prefill-fp8"], True),
                                            (False, [], None), (False, ["--no-prefill-fp8"], False)])
def test_the_prompt_precision_is_set_before_loading_and_shown(tmp_path, monkeypatch, capsys, fast, flags, fp8):
    """The switch is set before the engine loads (None: the default); a checkpoint without an FP8 prompt kernel
    refuses the flag by name and serves bf16 prompts otherwise."""

    import tensorfold.cuda.server as server
    from tensorfold.cuda import prompt_precision

    asked = prompt_precision.FP8_BY_DEFAULT if fp8 is None else fp8
    seen = []

    def engine(*a, **k):
        seen.append(prompt_precision.fp8())
        return SimpleNamespace(max_len=8192, w=SimpleNamespace(fast_prefill=fast))

    family = _family(cuda_engine=engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=8185))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"] + flags)
    try:
        if not fast and flags == ["--prefill-fp8"]:
            with pytest.raises(ValueError, match="no FP8 kernel"):
                cli._serve_cuda(args, family, tmp_path, 8192)
        else:
            assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
            shown = "FP8 activations" if asked and fast else "bf16 activations"
            assert f"prompts: {shown}" in capsys.readouterr().out
        assert seen == [asked]
    finally:
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)


def test_serve_parses_the_kv_cache_flag():
    plain = cli.build_parser().parse_args(["serve", "owner/model"])
    assert plain.kv_dtype == "bf16"                        # the cache stays bf16 unless it is asked for
    assert cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "int8"]).kv_dtype == "int8"
    assert cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "int4"]).kv_dtype == "int4"
    assert cli.build_parser().parse_args(["serve", "owner/model", "--mtp-confidence", "0.6"]).mtp_confidence == 0.6
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "fp8"])


def test_kv_dtype_is_refused_by_the_families_that_do_not_declare_it(tmp_path):
    """``CUDA_KV_DTYPES`` is the gate: a family that does not list a dtype is refused before anything loads."""

    from tensorfold.families import glm5_next, nemotron_h

    for module in (glm5_next, nemotron_h):
        args = argparse.Namespace(tp=1, rank=0, master="", master_port=29551, no_drafts=True, drafter="none",
                                  mtp_drafts=None, name="", model=str(tmp_path), kv_dtype="int8")
        with pytest.raises(ValueError, match="KV cache, not --kv-dtype int8"):
            cli._check_serve_options(args, SimpleNamespace(title=module.TITLE, package=module), "cuda")


@pytest.mark.parametrize("flags,backend,family,message", [
    (["--kv-dtype", "int8"], "mlx", "glm5_next", "MLX path caches keys and values as bf16"),
    (["--kv-dtype", "int4"], "cuda", "glm5_next", "KV cache, not --kv-dtype int4"),
    (["--kv-dtype", "int8"], "cuda", "nemotron_h", "KV cache, not --kv-dtype int8"),
    (["--mtp-confidence", "0.6"], "cuda", "glm5_next", "on CUDA has no such rule"),
    (["--prefill-fp8"], "cuda", "nemotron_h", "on CUDA has none"),
    (["--prefill-fp8"], "cuda", "glm5_next", "on CUDA has none"),
])
def test_cache_and_confidence_options_are_refused_before_any_download(tmp_path, monkeypatch, flags, backend, family,
                                                                      message):
    """Every family and backend answers ``--kv-dtype``, ``--mtp-confidence`` and ``--prefill-fp8``: served as asked,
    or refused by name before a weight moves; none ignores them."""

    import importlib

    from tensorfold import families, hub

    module = importlib.import_module(f"tensorfold.families.{family}")
    found = SimpleNamespace(title=module.TITLE, package=module, model_type=family)
    monkeypatch.setattr(families, "detect", lambda path: found)
    monkeypatch.setattr(cli, "_backend", lambda choice, fam: backend)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights were fetched before the refusal"))
    monkeypatch.setattr(families, "require_readable",
                        lambda *a: pytest.fail("the checkpoint was read before the refusal"))
    command = ["serve", str(tmp_path), "--no-update-check"] + flags
    with pytest.raises(ValueError, match=message):
        cli.cmd_serve(cli.build_parser().parse_args(command))


@pytest.mark.torch
def test_no_cuda_engine_serves_one_token_a_round_by_default(tmp_path, monkeypatch):
    """Everything on the lanes: GLM drafts with its MTP head and, when given, the DFlash2 drafter; --no-drafts and
    --mtp-drafts 0 without a drafter give the serial reference."""

    from tensorfold.families import glm5_next
    from tensorfold.families.glm5_next.cuda import engine as glm_engine

    made = []
    stub = lambda *a, **k: made.append(k) or SimpleNamespace(**k)      # noqa: E731
    monkeypatch.setattr(glm_engine, "GlmEngine", stub)

    # GLM: --mtp-drafts 0 with the DFlash2 drafter still drafts (DFlash2 alone); without it, the serial reference
    glm = dict(tp=2, master="192.0.2.10")
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), mtp_drafts=0, **glm).policy == "fc5:0.3"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=0, **glm).policy == "0"
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), **glm).policy == "auto"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=2, **glm).policy == "2"


@pytest.mark.parametrize("flag, streams", [(None, None), ("auto", None), ("1", None), ("4", 4)])
def test_cuda_parallel_is_one_request_at_a_time_unless_a_number_asks(tmp_path, monkeypatch, flag, streams):
    import tensorfold.cuda.server as server

    made = []
    engine = SimpleNamespace(context_window=4096)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=4096))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"] + (["--parallel", flag] if flag else [])
    assert cli._serve_cuda(cli.build_parser().parse_args(command), family, tmp_path, 4096) == 0
    assert made[0].get("parallel") == streams


@pytest.mark.parametrize("declares, expected", [(True, 6), (False, None)])
def test_checkpoint_slots_reach_a_cuda_engine_only_when_its_family_declares_them(tmp_path, monkeypatch, declares,
                                                                                    expected):
    import tensorfold.cuda.server as server

    made = []
    engine = SimpleNamespace(context_window=4096)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine,
                     **({"CUDA_CHECKPOINT_SLOTS": True} if declares else {}))
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=4096))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--parallel", "2", "--checkpoint-slots", "6"]
    assert cli._serve_cuda(cli.build_parser().parse_args(command), family, tmp_path, 4096) == 0
    assert made[0].get("checkpoint_slots") == expected
