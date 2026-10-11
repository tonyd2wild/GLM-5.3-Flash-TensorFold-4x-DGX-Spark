"""priority "background" on CUDA, as on the Mac: the one-at-a-time engine gives way to a foreground request that waits,
and a background reply decodes on later from its reply with the same tokens."""

import threading
import time

import pytest

from tensorfold.cuda.turns import Turns


def _until(check, seconds=10.0):
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.002)


def test_turns_give_the_engine_to_a_waiting_foreground_request_first():
    turns = Turns()
    turns.take(False)
    got = []
    back = threading.Thread(target=lambda: (turns.take(True), got.append("back"), turns.give()))
    back.start()
    time.sleep(0.05)
    front = threading.Thread(target=lambda: (turns.take(False), got.append("front"), turns.give()))
    front.start()
    time.sleep(0.05)
    assert turns.wanted()
    turns.give()
    back.join()
    front.join()
    assert got == ["front", "back"]


class Steady:
    """One request at a time, a token a round (position-keyed, so a continuation writes the same ones)."""

    eos = (0,)
    tp = 1

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        for i in range(max_tokens):
            time.sleep(0.004)
            if on_tokens([100 + (len(prompt) + i) % 50]):
                break
        return {"rounds": max_tokens}


def _body(content, count, background):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": count, "return_token_ids": True}
    return {**body, "priority": "background"} if background else body


def test_one_at_a_time_a_background_reply_yields_between_rounds(tmp_path):
    from tests.test_cuda_tool_choice import app_for

    alone = app_for(tmp_path, Steady())
    want = alone.run(_body("tell me a story", 60, True), True, lambda d: True)["stats"]["token_ids"]
    engine = Steady()
    app = app_for(tmp_path, engine)
    got, deltas = {}, []
    worker = threading.Thread(target=lambda: got.update(back=app.run(_body("tell me a story", 60, True), True,
                                                                     lambda d: deltas.append(d) or True)))
    worker.start()
    _until(lambda: len(deltas) >= 5)
    got["front"] = app.run(_body("hi", 3, False), True, lambda d: True)
    assert worker.is_alive()                                  # the foreground reply came first
    worker.join()
    assert got["back"]["stats"]["token_ids"] == want          # the same tokens as alone
    first, front, rest = engine.prompts                       # it stopped, the foreground ran, it replayed
    assert len(front) < len(first) and rest == first


def test_a_session_title_request_is_background(tmp_path):
    from tests.test_cuda_tool_choice import app_for

    engine = Steady()
    app = app_for(tmp_path, engine)
    seen = []
    real = app._turns().take
    app.turns.take = lambda background, cancelled=None: (seen.append(background), real(background, cancelled))[1]
    title = [{"role": "system", "content": "Write a short title for this conversation."},
             {"role": "user", "content": "hello"}]
    for messages in (title, [{"role": "user", "content": "hello"}]):
        app.run({"messages": messages, "max_tokens": 2}, True, lambda d: True)
    assert seen == [True, False]


class Shifting:
    """An engine whose tokens depend on where its run started: a run resumed from prompt + reply writes other tokens
    than the run it continues, so only a replay from the same prompt gives the reply alone."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, count, feed):
        self.prompts.append(list(prompt))
        for i in range(count):
            if feed([1000 + (7 * len(prompt) + i) % 50]):      # a longer prompt shifts every token
                break
        return {"rounds": count}


class Once:
    """A replay gate that cuts once, after ``at`` tokens."""

    replay = True

    def __init__(self, at):
        self.at, self.seen, self.done = at, 0, False

    def cut(self, tokens):
        if self.done or self.seen + len(tokens) < self.at:
            return None
        self.done = True
        return self.at - self.seen, []

    def observe(self, token):
        self.seen += 1


def test_a_yield_replays_from_the_same_prompt_and_sends_each_token_once():
    from tensorfold.engine.call_gate import generate_gated

    alone, sent = Shifting(), []
    generate_gated(alone.generate, [7] * 5, 12, [], lambda new: sent.extend(new) or False)
    want, sent = list(sent), []
    engine = Shifting()
    generate_gated(engine.generate, [7] * 5, 12, [Once(4)], lambda new: sent.extend(new) or False)
    assert sent == want and engine.prompts == [[7] * 5, [7] * 5]


def test_a_replay_that_differs_from_what_it_sent_fails():
    from tensorfold.engine.call_gate import generate_gated

    runs = []

    def generate(prompt, count, feed):
        runs.append(1)
        for i in range(count):
            if feed([len(runs) * 100 + i]):                   # the second run writes other tokens
                break

    with pytest.raises(RuntimeError, match="replay"):
        generate_gated(generate, [1, 2], 8, [Once(3)], lambda new: False)
