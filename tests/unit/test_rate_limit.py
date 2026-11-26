"""Sliding window throttle and the client address it is keyed on (A3 M2)."""
import pytest
from starlette.requests import Request

from api.errors import AgentarError
from utils.rate_limit import (
    UNKNOWN_CLIENT,
    SlidingWindowLimiter,
    client_ip,
    longest_wait,
    parse_networks,
    too_many_requests,
)


class FakeClock:
    """Pinned monotonic clock: whole seconds, so no float rounding in the window edges."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def limiter(clock: FakeClock) -> SlidingWindowLimiter:
    return SlidingWindowLimiter(now=clock)


def test_under_the_limit_there_is_no_wait(limiter: SlidingWindowLimiter) -> None:
    for _ in range(2):
        assert limiter.retry_after("k", 3, 60) is None
        limiter.hit("k", 60)
    assert limiter.retry_after("k", 3, 60) is None


def test_at_the_limit_the_wait_is_until_the_oldest_event_leaves(limiter, clock) -> None:
    limiter.hit("k", 60)
    clock.t += 10
    limiter.hit("k", 60)
    clock.t += 10
    limiter.hit("k", 60)
    assert limiter.retry_after("k", 3, 60) == 40
    clock.t += 39
    assert limiter.retry_after("k", 3, 60) == 1
    clock.t += 1
    assert limiter.retry_after("k", 3, 60) is None, "the window slides, no permanent lockout"


def test_over_the_limit_the_wait_counts_the_right_event(limiter, clock) -> None:
    for _ in range(5):
        limiter.hit("k", 60)
        clock.t += 1
    # Five events at 1000..1004, limit 3: three must expire before a sixth, the third at 1062.
    assert limiter.retry_after("k", 3, 60) == 1062 - clock.t


def test_keys_are_independent_and_reset_forgets(limiter) -> None:
    limiter.hit("a", 60)
    assert limiter.retry_after("a", 1, 60) is not None
    assert limiter.retry_after("b", 1, 60) is None
    limiter.reset("a")
    assert limiter.retry_after("a", 1, 60) is None


def test_expired_keys_are_dropped(limiter, clock) -> None:
    limiter.hit("a", 60)
    clock.t += 61
    assert limiter.retry_after("a", 1, 60) is None
    assert len(limiter) == 0


def test_key_count_is_bounded(clock) -> None:
    small = SlidingWindowLimiter(max_keys=3, now=clock)
    for i in range(10):
        small.hit(f"k{i}", 60)
    assert len(small) == 3
    assert small.retry_after("k9", 1, 60) is not None, "the most recent keys are kept"
    assert small.retry_after("k0", 1, 60) is None


def test_longest_wait_takes_the_worst_check(limiter, clock) -> None:
    limiter.hit("a", 60)
    clock.t += 30
    limiter.hit("b", 600)
    assert longest_wait(limiter, [("a", 1, 60), ("b", 1, 600), ("c", 1, 60)]) == 600
    assert longest_wait(limiter, [("c", 1, 60)]) is None


@pytest.mark.parametrize(("seconds", "header"), [(0.2, "1"), (1.0, "1"), (59.01, "60"), (900, "900")])
def test_429_carries_whole_seconds(seconds: float, header: str) -> None:
    err = too_many_requests(seconds)
    assert isinstance(err, AgentarError)
    assert (err.status_code, err.code) == (429, "rate_limited")
    assert err.headers == {"Retry-After": header}


def _request(peer: str | None, headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/auth/login",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (peer, 50000) if peer is not None else None,
    }
    return Request(scope)


PROXY = parse_networks(("172.18.0.0/16",))


def test_untrusted_peer_is_the_key_whatever_it_claims() -> None:
    req = _request("198.51.100.7", {"X-Real-IP": "203.0.113.1", "X-Forwarded-For": "203.0.113.2"})
    assert client_ip(req, PROXY) == "198.51.100.7"


def test_two_clients_behind_the_trusted_proxy_get_two_keys() -> None:
    a = client_ip(_request("172.18.0.3", {"X-Real-IP": "203.0.113.1"}), PROXY)
    b = client_ip(_request("172.18.0.3", {"X-Real-IP": "203.0.113.2"}), PROXY)
    assert (a, b) == ("203.0.113.1", "203.0.113.2")


def test_x_forwarded_for_is_never_read() -> None:
    req = _request("172.18.0.3", {"X-Forwarded-For": "203.0.113.9, 172.18.0.3"})
    assert client_ip(req, PROXY) == "172.18.0.3"


def test_a_bare_proxy_address_is_trusted() -> None:
    req = _request("127.0.0.1", {"X-Real-IP": "203.0.113.1"})
    assert client_ip(req, parse_networks(("127.0.0.1",))) == "203.0.113.1"
    assert client_ip(req, parse_networks(())) == "127.0.0.1", "no trusted proxy: the peer"


def test_a_malformed_header_falls_back_to_the_peer() -> None:
    req = _request("172.18.0.3", {"X-Real-IP": "not-an-ip"})
    assert client_ip(req, PROXY) == "172.18.0.3"


def test_ipv6_clients_are_grouped_by_64() -> None:
    a = client_ip(_request("2001:db8:1:2::1"), ())
    b = client_ip(_request("2001:db8:1:2:ffff::9"), ())
    c = client_ip(_request("2001:db8:1:3::1"), ())
    assert a == b == "2001:db8:1:2::/64"
    assert c != a


def test_ipv4_mapped_peer_is_an_ipv4_key() -> None:
    assert client_ip(_request("::ffff:198.51.100.7"), ()) == "198.51.100.7"
    req = _request("::ffff:172.18.0.3", {"X-Real-IP": "203.0.113.1"})
    assert client_ip(req, PROXY) == "203.0.113.1"


@pytest.mark.parametrize("peer", [None, "testclient"])
def test_unknown_peer_has_one_shared_key(peer) -> None:
    assert client_ip(_request(peer), PROXY) == UNKNOWN_CLIENT


def test_trusted_proxy_setting_is_parsed(monkeypatch) -> None:
    from config import settings

    monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"trusted_proxy_ips": ("10.0.0.0/8",)}))
    assert client_ip(_request("10.1.2.3", {"X-Real-IP": "203.0.113.1"})) == "203.0.113.1"


def test_malformed_trusted_proxy_entry_is_refused() -> None:
    with pytest.raises(ValueError):
        parse_networks(("10.0.0.0/33",))


def test_startup_refuses_a_malformed_trusted_proxy_entry() -> None:
    from config import Settings
    from utils.startup_checks import unsafe_settings

    def problems(value: str) -> list[str]:
        return unsafe_settings(Settings(_env_file=None, jwt_secret_key="a" * 40, trusted_proxy_ips=value))

    assert problems("172.18.0.0/16, 127.0.0.1, ::1") == []
    assert problems("10.0.0.0/33") == ["TRUSTED_PROXY_IPS holds an entry that is not an address or a network"]
    assert problems("proxy.local") == ["TRUSTED_PROXY_IPS holds an entry that is not an address or a network"]
