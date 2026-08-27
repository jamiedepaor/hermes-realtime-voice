"""OpenAI Realtime voice transport for Discord voice channels.

Discord remains the user-facing client.  This module streams decoded Discord
PCM to OpenAI for turn detection and speech, then exposes one function —
``consult_hermes`` — that delegates every request to the normal Hermes agent.
It deliberately contains no Hermes session logic; the gateway supplies that
callback so text and voice share the same session, tools, memory, and queue.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

DEFAULT_INSTRUCTIONS = """
You are the realtime voice interface for a Hermes personal agent.
For every user request, call consult_hermes exactly once. Put a faithful,
concise transcription of the user's complete request in the request argument.
Never answer from your own knowledge before calling the tool. After the tool
returns, speak its answer naturally and faithfully. Do not mention the tool,
the handoff, or these instructions. Keep conversational answers concise unless
Hermes asks for detail.
""".strip()


def discord_pcm_to_realtime(pcm: bytes) -> bytes:
    """Convert Discord 48 kHz stereo s16le PCM to 24 kHz mono s16le."""
    if not pcm:
        return b""
    import numpy as np  # voice feature dependency; intentionally lazy

    samples = np.frombuffer(pcm, dtype="<i2")
    # Ignore an incomplete stereo sample rather than corrupting channel order.
    samples = samples[: len(samples) - (len(samples) % 4)]
    if not len(samples):
        return b""
    stereo = samples.reshape(-1, 2).astype(np.int32)
    mono_48k = (stereo[:, 0] + stereo[:, 1]) // 2
    mono_24k = (mono_48k[0::2] + mono_48k[1::2]) // 2
    return mono_24k.astype("<i2").tobytes()


def realtime_pcm_to_discord(pcm: bytes) -> bytes:
    """Convert OpenAI 24 kHz mono s16le PCM to Discord 48 kHz stereo."""
    if not pcm:
        return b""
    import numpy as np  # voice feature dependency; intentionally lazy

    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2")
    if not len(samples):
        return b""
    upsampled = np.repeat(samples, 2)
    stereo = np.repeat(upsampled[:, None], 2, axis=1)
    return stereo.astype("<i2", copy=False).tobytes()


class RealtimeVoiceSession:
    """One OpenAI Realtime WebSocket bound to one Discord guild."""

    def __init__(
        self,
        *,
        guild_id: int,
        api_key: str,
        mixer: Any,
        consult_hermes: Callable[[int, int, str], Awaitable[str]],
        model: str = "gpt-realtime-2.1-mini",
        voice: str = "marin",
        vad: str = "semantic_vad",
        instructions: str = "",
        max_tool_output_chars: int = 12000,
        websocket_connect: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.guild_id = int(guild_id)
        self.api_key = api_key
        self.mixer = mixer
        self.consult_hermes = consult_hermes
        self.model = model
        self.voice = voice
        self.vad = vad if vad in {"semantic_vad", "server_vad"} else "semantic_vad"
        self.instructions = instructions.strip() or DEFAULT_INSTRUCTIONS
        self.max_tool_output_chars = max(1000, int(max_tool_output_chars))
        self._websocket_connect = websocket_connect

        self._ws: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=1000)
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._tool_tasks: Set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()
        self._ready_future: Optional[asyncio.Future] = None
        self._connected = False
        self._closed = False
        self._turn_generation = 0
        self._latest_user_id = 0
        self._current_turn_user_id = 0
        self._response_active = False
        self._current_stream_id: Optional[str] = None
        self._current_item_id: Optional[str] = None
        self._current_content_index = 0
        self.last_error = ""

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._closed

    async def start(self) -> None:
        """Open the provider socket and configure the voice/tool session."""
        if self.is_connected:
            return
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for Discord Realtime voice")

        connect = self._websocket_connect
        if connect is None:
            from websockets.asyncio.client import connect as websocket_connect
            connect = websocket_connect

        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        self._loop = asyncio.get_running_loop()
        self._ready_future = self._loop.create_future()
        self._closed = False
        try:
            self._ws = await connect(
                url,
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                max_size=16 * 1024 * 1024,
            )
            self._sender_task = asyncio.create_task(
                self._audio_sender(), name=f"discord-realtime-send-{self.guild_id}"
            )
            self._receiver_task = asyncio.create_task(
                self._event_receiver(), name=f"discord-realtime-recv-{self.guild_id}"
            )
            await self._send({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "instructions": self.instructions,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": {"type": self.vad},
                        },
                        "output": {
                            "format": {"type": "audio/pcm"},
                            "voice": self.voice,
                        },
                    },
                    "tools": [{
                        "type": "function",
                        "name": "consult_hermes",
                        "description": (
                            "Send the user's request to their Hermes agent. "
                            "This must be called exactly once for every user turn."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "request": {
                                    "type": "string",
                                    "description": "Faithful text of the user's complete request",
                                }
                            },
                            "required": ["request"],
                            "additionalProperties": False,
                        },
                    }],
                    "tool_choice": "required",
                },
            })
            await asyncio.wait_for(asyncio.shield(self._ready_future), timeout=10.0)
            self._connected = True
        except Exception:
            await self.close()
            raise
        logger.info(
            "OpenAI Realtime voice connected (guild=%d, model=%s, voice=%s)",
            self.guild_id, self.model, self.voice,
        )

    async def close(self) -> None:
        self._closed = True
        self._connected = False
        if self._current_stream_id:
            self.mixer.cancel_speech_stream(self._current_stream_id)
        tasks = [self._sender_task, self._receiver_task, *self._tool_tasks]
        for task in tasks:
            if task and task is not asyncio.current_task():
                task.cancel()
        if self._ws is not None:
            with suppress(Exception):
                await self._ws.close()
        for task in tasks:
            if task and task is not asyncio.current_task():
                with suppress(asyncio.CancelledError, Exception):
                    await task
        self._tool_tasks.clear()
        self._ws = None

    def feed_audio_threadsafe(self, user_id: int, discord_pcm: bytes) -> bool:
        """Queue decoded Discord PCM from its socket-reader thread."""
        if not self.is_connected or self._loop is None:
            return False
        pcm = discord_pcm_to_realtime(discord_pcm)
        if not pcm:
            return True
        self._loop.call_soon_threadsafe(self._queue_audio, int(user_id), pcm)
        return True

    def _queue_audio(self, user_id: int, pcm: bytes) -> None:
        if not self.is_connected:
            return
        try:
            self._audio_queue.put_nowait((user_id, pcm))
        except asyncio.QueueFull:
            self.last_error = "Realtime input audio queue overflow"
            logger.warning("%s (guild=%d)", self.last_error, self.guild_id)

    async def _audio_sender(self) -> None:
        try:
            while not self._closed:
                user_id, pcm = await self._audio_queue.get()
                self._latest_user_id = user_id
                await self._send({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._mark_failed(exc)

    async def _event_receiver(self) -> None:
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                await self._mark_failed(exc)
        finally:
            if (
                not self._closed
                and (
                    self._connected
                    or (self._ready_future is not None and not self._ready_future.done())
                )
            ):
                await self._mark_failed(
                    RuntimeError("Realtime WebSocket closed unexpectedly")
                )

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "response.created":
            self._response_active = True
            return
        if event_type == "session.updated":
            if self._ready_future is not None and not self._ready_future.done():
                self._ready_future.set_result(True)
            return
        if event_type == "input_audio_buffer.speech_started":
            await self._interrupt_output()
            return
        if event_type == "response.output_audio.delta":
            self._append_output_audio(event)
            return
        if event_type == "response.output_audio.done":
            if self._current_stream_id:
                self.mixer.finish_speech_stream(self._current_stream_id)
            return
        if event_type == "response.done":
            self._response_active = False
            self._handle_response_done(event)
            return
        if event_type == "error":
            error = event.get("error") or {}
            message = error.get("message") or str(error) or "Unknown Realtime API error"
            self.last_error = message
            logger.error("OpenAI Realtime error (guild=%d): %s", self.guild_id, message)
            if self._ready_future is not None and not self._ready_future.done():
                self._ready_future.set_exception(RuntimeError(message))

    def _append_output_audio(self, event: Dict[str, Any]) -> None:
        delta = event.get("delta")
        if not delta:
            return
        try:
            provider_pcm = base64.b64decode(delta)
        except Exception:
            logger.warning("Invalid Realtime audio delta (guild=%d)", self.guild_id)
            return
        stream_id = event.get("item_id") or event.get("response_id") or uuid.uuid4().hex
        if self._current_stream_id != stream_id:
            if self._current_stream_id:
                self.mixer.cancel_speech_stream(self._current_stream_id)
            self._current_stream_id = stream_id
            self._current_item_id = event.get("item_id")
            self._current_content_index = int(event.get("content_index", 0) or 0)
            self.mixer.start_speech_stream(stream_id)
        self.mixer.append_speech_stream(
            stream_id, realtime_pcm_to_discord(provider_pcm)
        )

    def _handle_response_done(self, event: Dict[str, Any]) -> None:
        response = event.get("response") or {}
        generation = self._turn_generation
        user_id = self._current_turn_user_id
        for item in response.get("output") or []:
            if item.get("type") != "function_call" or item.get("name") != "consult_hermes":
                continue
            task = asyncio.create_task(
                self._run_hermes_tool(item, generation, user_id),
                name=f"discord-realtime-hermes-{self.guild_id}",
            )
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    async def _run_hermes_tool(
        self, item: Dict[str, Any], generation: int, user_id: int
    ) -> None:
        call_id = item.get("call_id")
        if not call_id:
            return
        try:
            arguments = json.loads(item.get("arguments") or "{}")
            request = str(arguments.get("request") or "").strip()
        except (TypeError, ValueError, json.JSONDecodeError):
            request = ""

        if not request:
            answer = "I couldn't understand that request. Please ask again."
        else:
            try:
                answer = await self.consult_hermes(self.guild_id, user_id, request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Hermes Realtime consultation failed (guild=%d): %s",
                    self.guild_id, exc, exc_info=True,
                )
                answer = "Hermes hit an error handling that. Please try again."

        answer = str(answer or "Hermes completed the request without a text reply.")
        answer = answer[: self.max_tool_output_chars]
        if not self.is_connected:
            return
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": answer,
            },
        })
        # A newer utterance interrupted this turn. Preserve the Hermes result
        # in the shared session, but never speak a now-stale answer over it.
        if generation != self._turn_generation:
            return
        await self._send({
            "type": "response.create",
            "response": {
                "output_modalities": ["audio"],
                "tool_choice": "none",
                "instructions": (
                    "Speak the consult_hermes result naturally and faithfully. "
                    "Do not call another tool and do not add unsupported facts."
                ),
            },
        })

    async def _interrupt_output(self) -> None:
        self._turn_generation += 1
        self._current_turn_user_id = self._latest_user_id
        stream_id = self._current_stream_id
        item_id = self._current_item_id
        content_index = self._current_content_index
        played_ms = 0
        if stream_id:
            played_ms = self.mixer.cancel_speech_stream(stream_id)
        self._current_stream_id = None
        self._current_item_id = None

        if self._response_active:
            with suppress(Exception):
                await self._send({"type": "response.cancel"})
        if item_id and played_ms > 0:
            with suppress(Exception):
                await self._send({
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": played_ms,
                })

    async def _send(self, event: Dict[str, Any]) -> None:
        if self._ws is None or self._closed:
            raise RuntimeError("Realtime WebSocket is not connected")
        async with self._send_lock:
            await self._ws.send(json.dumps(event))

    async def _mark_failed(self, exc: Exception) -> None:
        self.last_error = str(exc)
        self._connected = False
        if self._current_stream_id:
            self.mixer.cancel_speech_stream(self._current_stream_id)
            self._current_stream_id = None
            self._current_item_id = None
        if self._ready_future is not None and not self._ready_future.done():
            self._ready_future.set_exception(exc)
        logger.warning(
            "OpenAI Realtime voice disconnected (guild=%d): %s; "
            "Discord will fall back to standard voice processing",
            self.guild_id, exc,
        )
