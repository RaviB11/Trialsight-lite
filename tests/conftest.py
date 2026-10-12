import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_studies.json"


@pytest.fixture
def synthetic_studies():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["studies"]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self.url = "https://example.test"

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Returns queued responses; records calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_response():
    return FakeResponse


@pytest.fixture
def fake_session():
    return FakeSession
