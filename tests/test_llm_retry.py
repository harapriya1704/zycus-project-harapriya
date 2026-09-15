"""Tests for rate-limit retry helpers (:mod:`src.llm_retry`)."""

from __future__ import annotations

import httpx
import openai
import pytest
from src.llm_retry import (
    _retry_after_seconds,
    _sleep_with_countdown,
    describe_error,
    invoke_with_retry,
)


class _RateLimit(Exception):
    status_code = 429


class _Bursty:
    """Fails with HTTP 429 the first ``failures`` calls, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def invoke(self, messages) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise _RateLimit("slow down")
        return "ok"


def test_sleep_with_countdown_splits_long_waits(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    _sleep_with_countdown(65.0, what="for TPM window reset")
    # 30s tick + 30s tick + 5s remainder, never blocking silently.
    assert sleeps == [30.0, 30.0, 5.0]


def test_success_on_first_call(monkeypatch) -> None:
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda _: None)
    llm = _Bursty(failures=0)
    assert invoke_with_retry(llm, ["hi"]) == "ok"
    assert llm.calls == 1


def test_retries_rate_limit_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    llm = _Bursty(failures=2)
    assert invoke_with_retry(llm, ["hi"], base_delay=1.0, max_delay=60.0) == "ok"
    assert llm.calls == 3
    assert sleeps == [1.0, 2.0]


def test_non_rate_limit_raises_immediately(monkeypatch) -> None:
    calls: list[str] = []
    sleeper = lambda _: None  # noqa: E731
    monkeypatch.setattr("src.llm_retry.time.sleep", sleeper)

    class _Boom(Exception):
        pass

    class _Failing:
        def invoke(self, messages) -> None:
            calls.append(messages)
            raise _Boom("nope")

    with pytest.raises(_Boom):
        invoke_with_retry(_Failing(), ["hi"], max_attempts=5)
    assert len(calls) == 1  # never retried


def test_rate_limit_budget_exhausted_raises_original(monkeypatch) -> None:
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda _: None)
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    llm = _Bursty(failures=99)
    with pytest.raises(_RateLimit):
        invoke_with_retry(llm, ["hi"], max_attempts=3)
    assert llm.calls == 3


class _WithBody(Exception):
    def __init__(self, message: str, body: object) -> None:
        super().__init__(message)
        self.body = body


class _TokenLimit(_WithBody):
    status_code = 413


def test_rate_limit_429_with_retry_after_emits_groq_pacing_warning(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    warnings_seen: list[str] = []

    class _Log:
        def warning(self, message, *args) -> None:
            warnings_seen.append(str(message))

        def error(self, message, *args) -> None:
            warnings_seen.append(str(message))

    monkeypatch.setattr("src.llm_retry.log", _Log())

    # Groq 429s carry a Retry-After header but no "tokens per minute" marker,
    # so _is_token_limit() is False here; the rate-limit path must still
    # surface the explicit pacing warning.
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    exc = httpx.HTTPStatusError(
        "Client error '429 Too Many Requests'",
        request=request,
        response=_httpx_response(429, {"retry-after": "30"}),
    )
    llm = _bursty_once(exc)
    assert invoke_with_retry(llm, ["hi"], max_attempts=3) == "ok"
    assert llm.calls == 2
    assert sleeps == [30.0]
    assert any("Groq TPM limit reached. Pacing request..." in line for line in warnings_seen)


def test_token_limit_retried_with_fixed_wait(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))

    class _Bursty:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages) -> str:
            self.calls += 1
            if self.calls == 1:
                raise _TokenLimit(
                    "Request too large",
                    {
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": "tokens per minute (TPM)",
                        }
                    },
                )
            return "ok"

    llm = _Bursty()
    assert invoke_with_retry(llm, ["hi"], token_reset_delay=60.0) == "ok"
    assert llm.calls == 2
    # The live countdown splits the 60s window into 30s ticks and re-sleeps the
    # remainder, so the total still equals token_reset_delay.
    assert sum(sleeps) == 60.0


def test_describe_error_includes_response_body() -> None:
    exc = _WithBody("broken", {"error": {"message": "model does not support images"}})
    desc = describe_error(exc)
    assert "_WithBody: broken" in desc
    assert "model does not support images" in desc


def test_describe_error_unwraps_cause() -> None:
    inner = _WithBody("inner boom", "body-text")
    outer = RuntimeError("outer failure")
    outer.__cause__ = inner
    desc = describe_error(outer)
    assert "outer failure" in desc
    assert "inner boom" in desc
    assert "body-text" in desc


def _httpx_response(status: int, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status, request=request, headers=headers or {})


def test_openai_rate_limit_error_retried_with_retry_after(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))

    exc = openai.RateLimitError(
        "Request was rejected due to a rate limit",
        response=_httpx_response(429, {"retry-after": "3"}),
        body={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
    )

    class _Bursty:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages) -> str:
            self.calls += 1
            if self.calls == 1:
                raise exc
            return "ok"

    llm = _Bursty()
    assert invoke_with_retry(llm, ["hi"]) == "ok"
    assert llm.calls == 2
    assert sleeps == [3.0]  # Retry-After honoured exactly


def test_httpx_http_status_error_429_backoff(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    exc = httpx.HTTPStatusError(
        "Client error '429 Too Many Requests'",
        request=request,
        response=_httpx_response(429),
    )

    class _Bursty:
        def __init__(self, failures: int) -> None:
            self.failures = failures
            self.calls = 0

        def invoke(self, messages) -> str:
            self.calls += 1
            if self.calls <= self.failures:
                raise exc
            return "ok"

    llm = _Bursty(failures=2)
    assert invoke_with_retry(llm, ["hi"], base_delay=1.0, max_delay=60.0) == "ok"
    assert sleeps == [1.0, 2.0]  # base_delay * 2^(attempt-1), no Retry-After header


def test_retry_after_wrapped_in_cause_is_honoured(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    inner = httpx.HTTPStatusError(
        "429",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        response=_httpx_response(429, {"retry-after": "7.5"}),
    )
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner

    class _Bursty:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages) -> str:
            self.calls += 1
            if self.calls == 1:
                raise outer
            return "ok"

    llm = _Bursty()
    assert invoke_with_retry(llm, ["hi"]) == "ok"
    assert llm.calls == 2
    assert sleeps == [7.5]


def test_retry_after_seconds_ignores_non_numeric_and_caps() -> None:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    assert (
        _retry_after_seconds(
            httpx.HTTPStatusError(
                "429",
                request=req,
                response=_httpx_response(429, {"retry-after": "Fri, 31 Dec 2049"}),
            ),
            cap=300.0,
        )
        is None
    )
    capped = _retry_after_seconds(
        httpx.HTTPStatusError(
            "429", request=req, response=_httpx_response(429, {"retry-after": "9000"})
        ),
        cap=300.0,
    )
    assert capped == 300.0


def _bursty_once(exc: BaseException):
    class _Bursty:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages) -> str:
            self.calls += 1
            if self.calls == 1:
                raise exc
            return "ok"

    return _Bursty()


def test_api_status_error_5xx_is_retried(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    exc = openai.APIStatusError(
        "503 Service Unavailable",
        response=_httpx_response(503),
        body={"error": {"message": "overloaded"}},
    )
    llm = _bursty_once(exc)
    assert invoke_with_retry(llm, ["hi"], base_delay=1.0, max_delay=60.0) == "ok"
    assert llm.calls == 2
    assert sleeps == [1.0]


def test_api_timeout_error_is_retried(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    exc = openai.APITimeoutError(request=request)
    llm = _bursty_once(exc)
    assert invoke_with_retry(llm, ["hi"], base_delay=1.0, max_delay=60.0) == "ok"
    assert sleeps == [1.0]


def test_httpx_request_error_is_retried(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    exc = httpx.TimeoutException("connection timed out")
    llm = _bursty_once(exc)
    assert invoke_with_retry(llm, ["hi"], base_delay=1.0, max_delay=60.0) == "ok"
    assert sleeps == [1.0]


def test_5xx_wrapped_in_langchain_exception_is_retried(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.llm_retry.random.uniform", lambda a, b: 0.0)

    inner = openai.APIStatusError(
        "503 Service Unavailable",
        response=_httpx_response(503),
        body={"error": {"message": "overloaded"}},
    )
    outer = RuntimeError("langchain wrapped the provider error")
    outer.__cause__ = inner
    llm = _bursty_once(outer)
    assert invoke_with_retry(llm, ["hi"], base_delay=1.0, max_delay=60.0) == "ok"
    assert sleeps == [1.0]


def test_api_status_error_400_is_not_retried(monkeypatch) -> None:
    monkeypatch.setattr("src.llm_retry.time.sleep", lambda _: None)
    exc = openai.APIStatusError(
        "400 Bad Request",
        response=_httpx_response(400),
        body={"error": {"message": "malformed payload"}},
    )
    llm = _bursty_once(exc)
    with pytest.raises(openai.APIStatusError):
        invoke_with_retry(llm, ["hi"], max_attempts=3)
    assert llm.calls == 1  # non-transient: never retried
