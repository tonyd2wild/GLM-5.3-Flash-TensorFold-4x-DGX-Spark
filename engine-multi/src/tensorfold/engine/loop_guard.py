"""A think block stuck repeating itself ends: the same gate hook as the thinking budget, one guard per request."""
from __future__ import annotations

from collections import Counter, deque
from typing import Sequence

WINDOW = 256      # committed think tokens a collapse has to fill
PERIOD = 16       # longest exact cycle (in tokens) that counts as a loop
DOMINANT = 0.5    # share of the window one token id may take before the block counts as collapsed


class LoopGuard:
    """Watches the committed tokens of one request while its think block is open. Once they collapse, the next
    round's first token becomes ``close`` (the thinking budget's close; ``ThinkBudget.cut``'s contract). Two collapses
    count:

    * an exact cycle of at most PERIOD tokens that held for WINDOW tokens in a row ("maybe maybe maybe ...");
    * one token id taking at least DOMINANT of the last WINDOW tokens, the cycle broken by other text ("the" in
      every other word of a degenerate block).

    Nothing is watched after ``think_end``. The fire latches until the close is committed; ``fires`` counts them.
    The guard holds no engine state and is made per request, so concurrent lanes are independent, and the server's
    gate runs on rank 0 as the budget's does."""

    def __init__(self, close: Sequence[int], think_end: int) -> None:
        self.close, self.think_end = [int(t) for t in close], int(think_end)
        self.open, self.fired, self.fires = True, False, 0
        self.recent: deque[int] = deque(maxlen=WINDOW)
        self.counts: Counter[int] = Counter()
        self.run = [0] * (PERIOD + 1)       # run[p]: consecutive tokens equal to the one p before
        self.last: deque[int] = deque(maxlen=PERIOD)

    def cut(self, tokens: Sequence[int]) -> tuple[int, list[int]] | None:
        """(0, close) once the block collapsed: the next round's tokens are not the reply, the close is."""

        if self.open and self.fired and len(tokens):
            return 0, list(self.close)
        return None

    def observe(self, token: int) -> None:
        token = int(token)
        if not self.open:
            return
        if token == self.think_end:
            self.open = False
            return
        if self.fired:
            return                           # the close's own tokens arrive here; they are not the model's
        for p in range(1, len(self.last) + 1):
            self.run[p] = self.run[p] + 1 if self.last[-p] == token else 0
        self.last.append(token)
        if len(self.recent) == WINDOW:
            self.counts[self.recent[0]] -= 1
        self.recent.append(token)
        self.counts[token] += 1
        if len(self.recent) == WINDOW and (any(r >= WINDOW for r in self.run) or self.counts[token] >= DOMINANT * WINDOW):
            self.fired = True
            self.fires += 1


__all__ = ["LoopGuard"]
