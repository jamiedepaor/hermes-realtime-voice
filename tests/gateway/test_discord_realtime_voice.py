"""Focused tests for the Discord ↔ OpenAI Realtime transport."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from plugins.platforms.discord.realtime_voice import (
    RealtimeVoiceSession,
    discord_pcm_to_realtime,
    realtime_pcm_to_discord,
)


class _Mixer:
    def __init__(self):
        self.started = []
        self.appended = []
        self.finished = []
        self.cancelled = []
        self.played_ms = 0

    def start_speech_stream(self, stream_id):
        self.started.append(stream_id)

    def append_speech_stream(self, stream_id, pcm):
        self.appended.append((stream_id, pcm))
        return True

    def finish_speech_stream(self, stream_id):
        self.finished.append(stream_id)
        return True

    def cancel_speech_stream(self, stream_id):
        self.cancelled.append(stream_id)
        return self.played_ms


class _WebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def close(self):
        return None


class _ConfigWebSocket(_WebSocket):
    def __init__(self):
        super().__init__()
        self.events = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.events.get()
        if event is None:
            raise StopAsyncIteration
        return json.dumps(event)

    async def send(self, payload):
        await super().send(payload)
        if self.sent[-1]["type"] == "session.update":
            await self.events.put({"type": "session.updated", "session": {}})

    async def close(self):
        await self.events.put(None)


def _session(consult=None):
    mixer = _Mixer()
    session = RealtimeVoiceSession(
        guild_id=123,
        api_key="test-key",
        mixer=mixer,
        consult_hermes=consult or AsyncMock(return_value="Hermes answer"),
    )
    session._ws = _WebSocket()
    session._connected = True
    return session, mixer


def test_discord_pcm_downmixes_and_downsamples():
    # Four 48 kHz stereo frames become two 24 kHz mono samples.
    source = np.array([
        [1000, 3000],
        [3000, 5000],
        [-1000, 1000],
        [1000, 3000],
    ], dtype=np.int16)
    converted = np.frombuffer(discord_pcm_to_realtime(source.tobytes()), dtype=np.int16)
    assert converted.tolist() == [3000, 1000]


def test_realtime_pcm_upsamples_and_duplicates_channels():
    source = np.array([100, -200], dtype=np.int16)
    converted = np.frombuffer(realtime_pcm_to_discord(source.tobytes()), dtype=np.int16)
    assert converted.reshape(-1, 2).tolist() == [
        [100, 100], [100, 100], [-200, -200], [-200, -200],
    ]


@pytest.mark.asyncio
async def test_start_configures_required_hermes_tool_and_pcm_audio():
    websocket = _ConfigWebSocket()
    connect = AsyncMock(return_value=websocket)
    mixer = _Mixer()
    session = RealtimeVoiceSession(
        guild_id=123,
        api_key="test-key",
        mixer=mixer,
        consult_hermes=AsyncMock(return_value="answer"),
        websocket_connect=connect,
    )

    await session.start()
    try:
        update = websocket.sent[0]
        assert update["type"] == "session.update"
        assert update["session"]["tool_choice"] == "required"
        assert update["session"]["tools"][0]["name"] == "consult_hermes"
        assert update["session"]["audio"]["input"]["format"] == {
            "type": "audio/pcm", "rate": 24000,
        }
        assert session.is_connected is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_audio_delta_streams_to_discord_mixer():
    session, mixer = _session()
    pcm = np.array([10, -10], dtype=np.int16).tobytes()
    await session._handle_event({
        "type": "response.output_audio.delta",
        "item_id": "item-1",
        "content_index": 0,
        "delta": base64.b64encode(pcm).decode("ascii"),
    })
    await session._handle_event({"type": "response.output_audio.done"})

    assert mixer.started == ["item-1"]
    assert mixer.appended[0][0] == "item-1"
    assert mixer.appended[0][1] == realtime_pcm_to_discord(pcm)
    assert mixer.finished == ["item-1"]


@pytest.mark.asyncio
async def test_function_call_uses_hermes_then_requests_audio():
    consult = AsyncMock(return_value="Use the Hermes result.")
    session, _ = _session(consult)
    session._current_turn_user_id = 456

    await session._handle_event({
        "type": "response.done",
        "response": {"output": [{
            "type": "function_call",
            "name": "consult_hermes",
            "call_id": "call-1",
            "arguments": json.dumps({"request": "check my calendar"}),
        }]},
    })
    await asyncio.gather(*session._tool_tasks)

    consult.assert_awaited_once_with(123, 456, "check my calendar")
    sent = session._ws.sent
    assert sent[0]["type"] == "conversation.item.create"
    assert sent[0]["item"]["output"] == "Use the Hermes result."
    assert sent[1]["type"] == "response.create"
    assert sent[1]["response"]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_barge_in_cancels_audio_and_truncates_conversation():
    session, mixer = _session()
    mixer.played_ms = 420
    session._response_active = True
    session._current_stream_id = "stream-1"
    session._current_item_id = "item-1"
    session._current_content_index = 2
    session._latest_user_id = 789

    await session._handle_event({"type": "input_audio_buffer.speech_started"})

    assert mixer.cancelled == ["stream-1"]
    assert session._current_turn_user_id == 789
    assert session._ws.sent == [
        {"type": "response.cancel"},
        {
            "type": "conversation.item.truncate",
            "item_id": "item-1",
            "content_index": 2,
            "audio_end_ms": 420,
        },
    ]
