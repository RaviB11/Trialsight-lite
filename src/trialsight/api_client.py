"""Minimal, polite client for the ClinicalTrials.gov API v2.

Design notes
- Rate limited client-side (config: requests_per_minute).
- Retries 429 / 5xx / network errors with exponential backoff, honouring Retry-After.
- Pagination follows the opaque ``nextPageToken`` returned by /studies.
- Clock and sleep are injectable so behaviour is testable without network access.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterator

import requests

from .config import ApiConfig

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "TrialSightLite/0.1 (personal open-data research project)"


class CTGovAPIError(RuntimeError):
    """Raised when the API returns an unrecoverable error."""


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = 60.0 / max(1, requests_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


class CTGovClient:
    def __init__(
        self,
        cfg: ApiConfig,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.session = session if session is not None else requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.setdefault("User-Agent", USER_AGENT)
        self._sleep = sleep
        self._limiter = RateLimiter(cfg.requests_per_minute, clock=clock, sleep=sleep)

    # ------------------------------------------------------------------ http
    def url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _backoff(attempt: int) -> float:
        return float(min(60, 2 ** attempt))

    @staticmethod
    def _retry_after(resp: Any) -> float | None:
        value = (getattr(resp, "headers", None) or {}).get("Retry-After")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self.url(path)
        attempts = self.cfg.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            self._limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout_seconds)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                log.warning("Network error on %s (attempt %d/%d): %s", url, attempt + 1, attempts, exc)
                if attempt + 1 < attempts:
                    self._sleep(self._backoff(attempt))
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise CTGovAPIError(f"Non-JSON response from {getattr(resp, 'url', url)}") from exc

            if resp.status_code in RETRYABLE_STATUS:
                last_error = CTGovAPIError(f"HTTP {resp.status_code} from {url}")
                log.warning("HTTP %s on %s (attempt %d/%d)", resp.status_code, url, attempt + 1, attempts)
                if attempt + 1 < attempts:
                    self._sleep(self._retry_after(resp) or self._backoff(attempt))
                continue

            body = (getattr(resp, "text", "") or "")[:300]
            raise CTGovAPIError(f"HTTP {resp.status_code} from {getattr(resp, 'url', url)}: {body}")

        raise CTGovAPIError(f"Giving up on {url} after {attempts} attempts") from last_error

    # ------------------------------------------------------------- endpoints
    def version(self) -> dict[str, Any]:
        return self.get_json("version")

    def studies_metadata(self) -> Any:
        return self.get_json("studies/metadata")

    def search_areas(self) -> Any:
        return self.get_json("studies/search-areas")

    def enums(self) -> Any:
        return self.get_json("studies/enums")

    def count(self, params: dict[str, Any]) -> int:
        p = dict(params, countTotal="true", pageSize=1)
        payload = self.get_json("studies", p)
        if "totalCount" not in payload:
            raise CTGovAPIError("Response did not include totalCount")
        return int(payload["totalCount"])

    def iter_study_pages(self, params: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
        """Yield (page_number, request_params, payload) until no nextPageToken."""
        base = dict(params)
        base.setdefault("format", "json")
        base["pageSize"] = self.cfg.page_size
        token: str | None = None
        seen_tokens: set[str] = set()
        page_no = 0
        while True:
            p = dict(base)
            if page_no == 0:
                p["countTotal"] = "true"
            if token:
                p["pageToken"] = token
            payload = self.get_json("studies", p)
            page_no += 1
            yield page_no, p, payload
            token = payload.get("nextPageToken")
            if not token:
                return
            if token in seen_tokens:
                raise CTGovAPIError(f"Pagination loop detected at page {page_no}")
            seen_tokens.add(token)
