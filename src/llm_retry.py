"""Rate-limit retry helpers for LLM API calls.

Wraps ``llm.invoke`` and lets *any* transient provider/transport failure retry
instead of killing a page/document in a bulk run:

- HTTP 429 (requests/min, RPM — ``openai.RateLimitError`` or an
  ``httpx.HTTPStatusError`` with status 429) is retried with a numeric
  ``Retry-After`` header honoured verbatim, else ``base_delay * 2 ** (attempt-1)``
  plus jitter.
- HTTP 413 ``rate_limit_exceeded`` / "tokens per minute" (TPM) is retried after
  a fixed ``TOKEN_RESET_DELAY`` so the rolling TPM window can clear.
- Other transient conditions are retried with the same back-off: 5xx +
  408/409 ``openai.APIStatusError``, ``openai.APITimeoutError`` / connection
  errors, and raw ``httpx.HTTPStatusError`` / ``httpx.RequestError``.

Nested ``__cause__`` / ``__context__`` chains are unwrapped so raw HTTPX
exceptions wrapped inside LangChain/OpenAI SDK errors never escape. Non-retryable
errors (auth failures, malformed requests) are re-raised immediately.
"""

from __future__ import annotations

import random
import time
from typing import Any

from src.logging_conf import get_logger

try:
    import httpx
    import openai
except ImportError:  # pragma: no cover - provider SDKs are optional dependencies
    httpx = None  # type: ignore[assignment]
    openai = None  # type: ignore[assignment]

log = get_logger(__name__)

#: Total call attempts made by default (first call + retries). 10 attempts give
#: long Groq TPM/RPM sliding windows several full back-off cycles to clear.
DEFAULT_MAX_ATTEMPTS = 10
#: Seconds to wait before the first retry.
BASE_DELAY = 1.0
#: Upper bound for the exponential back-off delay (seconds).
MAX_DELAY = 60.0
#: Wait used for per-minute token (TPM) limits — one full window.
TOKEN_RESET_DELAY = 60.0
#: Ceiling (seconds) applied to a provider-sent ``Retry-After`` value, so a
#: malformed/gigantic header can never stall the batch indefinitely.
MAX_RETRY_AFTER = 300.0
#: Non-429 HTTP statuses that are still transient enough to retry.
_TRANSIENT_STATUS = {408, 409, 425}
#: How deep nested ``__cause__`` / ``__context__`` chains are unwrapped before
#: giving up classification. LangChain normally wraps the OpenAI SDK directly
#: (2-3 levels); 8 leaves generous room for provider-gateway nesting.
_CHAIN_DEPTH = 8


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True when *exc* represents an HTTP 429 / rate-limit error.

    Catches ``openai.RateLimitError`` and ``httpx.HTTPStatusError`` with a 429
    response directly, plus any exception chain carrying a ``status_code`` of
    429 (``openai`` errors) or a ``response.status_code`` of 429. Walks up to
    ``_CHAIN_DEPTH`` levels of ``__cause__`` / ``__context__`` for wrapped
    exceptions.
    """
    if openai is not None and isinstance(exc, openai.RateLimitError):
        return True
    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
    for _ in range(_CHAIN_DEPTH):
        code = getattr(exc, "status_code", None)
        if isinstance(code, int) and code == 429:
            return True
        response = getattr(exc, "response", None)
        if response is not None and isinstance(getattr(response, "status_code", None), int):
            if response.status_code == 429:
                return True
        cause = exc.__cause__ if exc.__cause__ is not None else exc.__context__
        if cause is None:
            return False
        exc = cause  # type: ignore[assignment]
    return False


def _is_retryable(exc: BaseException) -> bool:
    """True when *exc* is a transient provider/transport failure worth retrying.

    Covers, at any depth of the cause chain:

    - rate limits: HTTP 429 (RPM) and 413/TPM (see :func:`_is_rate_limit` /
      :func:`_is_token_limit`),
    - the full OpenAI SDK surface: ``APITimeoutError`` / ``APIConnectionError``
      and ``APIStatusError`` with a transient status (5xx, 408, 409, 425),
    - raw HTTPX: ``RequestError`` (timeouts, connection resets) and
      ``HTTPStatusError`` with a transient status.
    """
    if _is_rate_limit(exc) or _is_token_limit(exc):
        return True

    current = exc
    for _ in range(_CHAIN_DEPTH):
        if httpx is not None and isinstance(current, httpx.RequestError):
            return True
        if openai is not None and isinstance(current, openai.APIConnectionError):
            return True
        if openai is not None and isinstance(current, openai.APIStatusError):
            status = getattr(current, "status_code", None)
            if _transient(status):
                return True
        if httpx is not None and isinstance(current, httpx.HTTPStatusError):
            response = getattr(current, "response", None)
            if response is not None and _transient(getattr(response, "status_code", None)):
                return True
        # Generic response objects (e.g. OpenAPI-fake responses in tests).
        response = getattr(current, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if _transient(status):
                return True
        cause = current.__cause__ if current.__cause__ is not None else current.__context__
        if cause is None:
            return False
        current = cause  # type: ignore[assignment]
    return False


def _transient(status: Any) -> bool:
    """A status is transient when it is 5xx or an explicit retryable 4xx."""
    if not isinstance(status, int):
        return False
    return status >= 500 or status in _TRANSIENT_STATUS


def _retry_after_seconds(exc: BaseException, cap: float = MAX_RETRY_AFTER) -> float | None:
    """Seconds to wait per a numeric ``Retry-After`` header, else ``None``.

    Prefers the header on the *nearest* level of the cause chain that exposes
    an HTTP response. A non-numeric (e.g. HTTP-date) value is ignored and the
    caller falls back to exponential back-off; ``cap`` guards against absurd
    values.
    """
    for _ in range(_CHAIN_DEPTH):
        response = getattr(exc, "response", None)
        if response is not None and isinstance(getattr(response, "status_code", None), int):
            headers = getattr(response, "headers", None)
            if headers is not None:
                raw = str(headers.get("retry-after", "") or "").strip()
                if raw:
                    try:
                        seconds = float(raw)
                    except ValueError:
                        seconds = None
                    if seconds is not None and seconds > 0:
                        return min(seconds, cap)
        cause = exc.__cause__ if exc.__cause__ is not None else exc.__context__
        if cause is None:
            return None
        exc = cause  # type: ignore[assignment]
    return None


def _is_token_limit(exc: BaseException) -> bool:
    """True when the provider reports a per-minute token (TPM) budget error.

    Groq surfaces these as HTTP 413 with ``error.code == 'rate_limit_exceeded'``
    and a "tokens per minute" message; other providers use ``tokens``/``tpm``
    codes (or 429 with a TPM body). Wrapped cause chains are unwrapped too.
    """
    for _ in range(_CHAIN_DEPTH):
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict):
                if str(inner.get("code", "")).lower() in {"rate_limit_exceeded", "tokens", "tpm"}:
                    return True
                if "tokens per minute" in str(inner.get("message", "")).lower():
                    return True
        if "tokens per minute" in str(exc).lower():
            return True
        cause = exc.__cause__ if exc.__cause__ is not None else exc.__context__
        if cause is None:
            return False
        exc = cause  # type: ignore[assignment]
    return False


def _truncate(text: str, limit: int = 2000) -> str:
    """Trim a response body so terminal logs do not blow up."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [truncated, {len(text)} chars]"


def describe_error(exc: BaseException) -> str:
    """Return a readable, terminal-safe description of a provider/API error.

    Includes the exception class, its message and — when the client exposes
    one — the raw response body (``openai`` API errors carry ``.body`` /
    ``.response.text``). Nested ``__cause__`` chains are unwrapped so the
    root provider message is never hidden.
    """
    detail = f"{type(exc).__name__}: {exc}"
    body = getattr(exc, "body", None)
    if body is None:
        response = getattr(exc, "response", None)
        if response is not None:
            body = getattr(response, "text", None)
    if body is not None:
        detail += f" | response body: {_truncate(str(body))}"
    cause = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    if cause is not None:
        detail += f" | caused by: {describe_error(cause)}"
    return detail


def invoke_with_retry(
    llm: Any,
    messages: Any,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    token_reset_delay: float = TOKEN_RESET_DELAY,
) -> Any:
    """Invoke *llm*, retrying on any transient provider/transport error.

    - HTTP 429 (``openai.RateLimitError`` / ``httpx.HTTPStatusError``): if the
      response carries a numeric ``Retry-After`` header, sleep exactly that
      long; otherwise wait ``base_delay * 2 ** (attempt - 1)`` plus jitter
      (exponential back-off), capped at ``max_delay``.
    - Per-minute token (TPM) limits (HTTP 413 ``rate_limit_exceeded`` or a
      "tokens per minute" body) wait a full ``token_reset_delay`` window.
    - Other transient errors — ``openai.APITimeoutError`` / connection errors,
      ``openai.APIStatusError`` with a 5xx/408/409/425 status, and raw
      ``httpx.RequestError`` / ``httpx.HTTPStatusError`` — retry with the same
      back-off. Nested ``__cause__`` chains are inspected so raw HTTPX status
      errors wrapped inside LangChain do not escape.

    Non-retryable exceptions (auth failures, malformed requests) are re-raised
    immediately so genuine problems still surface fast.

    Args:
        llm: a LangChain-compatible chat model exposing ``.invoke(messages)``.
        messages: the message payload to send.
        max_attempts: total call budget (first attempt included).
        base_delay: seconds to wait before the first 429 retry (grows
            geometrically with each retry).
        max_delay: ceiling for the growing back-off delay.
        token_reset_delay: fixed wait applied to TPM (413) limits.

    Returns:
        The raw result of ``llm.invoke(messages)``.

    Raises:
        The original exception when the retry budget is exhausted or when the
        error is not retryable.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            token_limit = _is_token_limit(exc)
            if not _is_retryable(exc):
                # Genuine failure (auth, malformed request): never mask it.
                raise
            if attempt >= max_attempts:
                log.error(
                    "LLM call failed after %s attempts; giving up: %s",
                    attempt,
                    exc,
                )
                raise

            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                sleep_for = retry_after
                log.warning(
                    "Transient LLM error, attempt %s/%s; Retry-After header "
                    "says %.0fs, waiting ...",
                    attempt,
                    max_attempts,
                    sleep_for,
                )
            elif token_limit:
                sleep_for = token_reset_delay
                log.warning(
                    "Token-per-minute (TPM) limit on LLM call, attempt %s/%s; "
                    "waiting %.0fs for the window to reset ...",
                    attempt,
                    max_attempts,
                    sleep_for,
                )
            else:
                exponential = base_delay * (2 ** (attempt - 1))
                sleep_for = min(exponential + random.uniform(0, exponential * 0.25), max_delay)
                log.warning(
                    "Transient LLM error, attempt %s/%s; "
                    "retrying in %.2fs ...",
                    attempt,
                    max_attempts,
                    sleep_for,
                )
            time.sleep(sleep_for)
    raise AssertionError("unreachable")
