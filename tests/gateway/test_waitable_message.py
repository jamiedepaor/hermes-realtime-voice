"""The internal waitable ingress keeps BasePlatformAdapter queue semantics."""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, **kwargs):
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_handle_message_and_wait_returns_final_text_after_session_cleanup():
    adapter = _Adapter(
        PlatformConfig(enabled=True, token="test"), Platform.DISCORD
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "Hermes final answer"

    adapter.set_message_handler(handler)
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            user_id="456",
            chat_type="channel",
        ),
    )
    session_key = build_session_key(event.source)

    result = await asyncio.wait_for(adapter.handle_message_and_wait(event), timeout=2)

    assert result == "Hermes final answer"
    assert session_key not in adapter._active_sessions
