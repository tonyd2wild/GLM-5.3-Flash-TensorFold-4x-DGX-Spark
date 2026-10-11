"""The floor the CUDA startup leaves free in a memory pool: a tenth of the pool, rounded down, and at least 4 GiB."""

from types import SimpleNamespace

from tensorfold.cuda import capacity

GIB = capacity.GIB


def card(free: int, total: int) -> SimpleNamespace:
    """A stand-in torch whose GPU reports ``free`` of ``total`` bytes."""
    return SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (free, total)))


def test_a_large_pool_keeps_a_tenth_rounded_down():
    pool = 121 * GIB + 7
    assert capacity.reserve_bytes(pool) == pool // 10
    assert capacity.reserve_bytes(128 * GIB) == 128 * GIB // 10


def test_a_small_pool_keeps_four_gib():
    for pool in (8 * GIB, 20 * GIB, 40 * GIB):
        assert capacity.reserve_bytes(pool) == 4 * GIB


def test_both_kinds_of_grant_pay_the_floor(monkeypatch):
    monkeypatch.delenv(capacity.LIMIT_ENV, raising=False)
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 121 * GIB, "MemAvailable": 110 * GIB})
    monkeypatch.setattr(capacity, "unified", lambda torch: True)          # one pool: the host's available memory
    assert capacity.available_bytes(card(100 * GIB, 121 * GIB)) == 110 * GIB - (121 * GIB) // 10
    monkeypatch.setattr(capacity, "unified", lambda torch: False)         # a card: its own free memory
    assert capacity.available_bytes(card(100 * GIB, 121 * GIB)) == 100 * GIB - (121 * GIB) // 10
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)               # with or without /proc/meminfo
    assert capacity.available_bytes(card(100 * GIB, 121 * GIB)) == 100 * GIB - (121 * GIB) // 10


def test_host_staging_keeps_two_gib(monkeypatch):
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 9 * GIB})
    assert capacity.host_stream_bytes() == 7 * GIB
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    assert capacity.host_stream_bytes() is None
