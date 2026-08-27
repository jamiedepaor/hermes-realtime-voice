import json
import time

import pytest

from plugins.platforms.discord.hermes_cloud_client import HermesCloudClient
from plugins.platforms.discord.hermes_cloud_client import HermesCloudError


def test_cloud_client_requires_https():
    with pytest.raises(ValueError, match="HTTPS"):
        HermesCloudClient("http://example.test")


@pytest.mark.asyncio
async def test_valid_cached_access_token_needs_no_refresh(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "accessToken": "cached-access",
                "refreshToken": "cached-refresh",
                "expiresAt": time.time() + 3600,
                "provider": "nous",
            }
        ),
        encoding="utf-8",
    )
    client = HermesCloudClient(
        "https://example.agents.nousresearch.com",
        token_file=str(token_file),
    )

    async def unexpected_request(*args, **kwargs):
        raise AssertionError("refresh should not be called")

    monkeypatch.setattr(client, "_json_request", unexpected_request)
    assert await client._access_token() == "cached-access"


class _FakeConnection:
    def __init__(self):
        self.requests = []
        self.closed = False

    async def request(self, method, params, timeout):
        self.requests.append((method, params, timeout))
        if method == "session.create":
            return {"session_id": "session-1"}
        return {}

    async def wait_event(self, predicate, timeout):
        event = {
            "type": "message.complete",
            "session_id": "session-1",
            "payload": {"text": "Hermes answer"},
        }
        assert predicate(event)
        return event

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_ask_creates_and_reuses_a_remote_session(monkeypatch):
    client = HermesCloudClient("https://example.agents.nousresearch.com")
    first = _FakeConnection()
    second = _FakeConnection()
    connections = iter([first, second])

    async def connect():
        return next(connections)

    monkeypatch.setattr(client, "_connect", connect)

    assert await client.ask("First", conversation="voice") == "Hermes answer"
    assert await client.ask("Second", conversation="voice") == "Hermes answer"
    assert first.requests[0][0] == "session.create"
    assert first.requests[1][0] == "prompt.submit"
    assert second.requests[0][0] == "prompt.submit"
    assert first.closed and second.closed


@pytest.mark.asyncio
async def test_ask_recreates_a_stale_remote_session(monkeypatch):
    client = HermesCloudClient("https://example.agents.nousresearch.com")
    client._sessions["voice"] = "stale-session"
    stale = _FakeConnection()
    replacement = _FakeConnection()

    async def stale_request(method, params, timeout):
        stale.requests.append((method, params, timeout))
        raise HermesCloudError("session not found")

    stale.request = stale_request
    connections = iter([stale, replacement])

    async def connect():
        return next(connections)

    monkeypatch.setattr(client, "_connect", connect)

    assert await client.ask("Retry", conversation="voice") == "Hermes answer"
    assert stale.requests[0][0] == "prompt.submit"
    assert replacement.requests[0][0] == "session.create"
    assert stale.closed and replacement.closed
