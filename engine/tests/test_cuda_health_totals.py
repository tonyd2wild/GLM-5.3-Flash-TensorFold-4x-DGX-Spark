"""The CUDA server's GET /health (``tensorfold.cuda.health``): live reply tokens while a request runs, then the
engine's own stats folded in once it returns, and a concurrent engine's stream counts."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.request import urlopen
import json

from pytest import importorskip

importorskip("jinja2")              # the CUDA server renders chat templates with it

from tensorfold.cuda import health
from tests.test_cuda_server_disconnect import MESSAGES, WAIT, PacedEngine, app_for, post, serving


class ReportingEngine(PacedEngine):
    """PacedEngine whose generate returns a lane engine's per-request stats."""

    def generate(self, *args, **kwargs):
        rounds = super().generate(*args, **kwargs)["rounds"]
        return dict(prefill_s=0.25, decode_s=0.5, rounds=rounds, drafted=3 * rounds, accepted=rounds, cached=2)


def read_health(port: int) -> dict:
    with urlopen(f"http://127.0.0.1:{port}/health", timeout=WAIT) as answer:
        assert answer.status == 200
        return json.load(answer)


def test_live_tokens_count_at_once_and_engine_stats_when_the_request_returns(tmp_path):
    engine = ReportingEngine(hold_at=2)                 # the reply pauses after its second token
    served = app_for(tmp_path, engine)
    served.context_window = 131072
    with serving(served) as port, ThreadPoolExecutor(max_workers=1) as client:
        idle = read_health(port)
        assert (idle["ok"], idle["backend"], idle["busy"]) == (True, "tensorfold", False)
        assert idle["requests_running"] == idle["requests_total"] == 0 and "streams" not in idle
        assert idle.get("context_length") == 131072
        reply = client.submit(post, port, {"messages": MESSAGES, "max_tokens": 4})
        assert engine.held.wait(WAIT), "the request never reached its pause"
        busy = read_health(port)
        assert busy["busy"] is True and busy["requests_running"] == 1
        assert busy["completion_tokens_total"] == 2                       # counted as they are written
        assert busy["prompt_tokens_total"] == busy["rounds_total"] == 0   # the engine's figures come at the end
        engine.release.set()
        status, text = reply.result(timeout=WAIT)
        assert status == 200, text[:300]
        done = read_health(port)
    assert done["busy"] is False and done["requests_running"] == 0 and done["requests_total"] == 1
    assert done["completion_tokens_total"] == 4 and done["prompt_tokens_total"] > 0
    assert (done["prefill_seconds_total"], done["decode_seconds_total"]) == (0.25, 0.5)
    assert (done["rounds_total"], done["drafted_total"], done["accepted_total"], done["cached_tokens_total"]) == \
        (4, 12, 4, 2)


def test_tokens_written_before_a_failure_still_count(tmp_path):
    class Breaks(PacedEngine):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            on_tokens([ord("o"), ord("k")])
            raise RuntimeError("the engine broke")

    with serving(app_for(tmp_path, Breaks())) as port:
        status, _ = post(port, {"messages": MESSAGES, "max_tokens": 4})
        after = read_health(port)
    assert status == 500
    assert (after["requests_running"], after["requests_total"], after["completion_tokens_total"]) == (0, 1, 2)
    assert after["rounds_total"] == 0


def test_a_scheduler_s_streams_are_reported_and_the_counters_are_per_app(tmp_path):
    engine = PacedEngine()
    engine.scheduler = SimpleNamespace(decoder=SimpleNamespace(streams={7: None, 8: None, 9: None}, filling=[]),
                                       max_streams=6)
    served = app_for(tmp_path, engine)
    assert health.of(served).snapshot(served)["streams"] == {"decoding": 3, "prefilling": 0, "max": 6}
    assert health.of(served) is health.of(served)


def test_an_app_without_an_engine_still_reports_idle():
    bare = SimpleNamespace(served="nothing-loaded")
    body = health.of(bare).snapshot(bare)
    assert body["ok"] is True and body["busy"] is False
    assert "streams" not in body and "context_length" not in body
