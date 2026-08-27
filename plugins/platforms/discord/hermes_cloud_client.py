"""Authenticated client for forwarding Discord voice turns to Hermes Cloud.

This is intentionally transport-only: it uses the same native OAuth and
gateway JSON-RPC endpoints as Hermes Desktop.  The Discord/OpenAI sidecar can
therefore live outside Nous Cloud while every request still runs inside the
user's existing hosted Hermes instance, with its normal tools and memory.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote

import httpx


class HermesCloudError(RuntimeError):
    """A safe, credential-free Hermes Cloud transport error."""


def _token_value(tokens: Dict[str, Any], camel: str, snake: str) -> str:
    return str(tokens.get(camel) or tokens.get(snake) or "")


class _GatewayConnection:
    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._events: asyncio.Queue[dict] = asyncio.Queue()
        self._receiver = asyncio.create_task(self._receive(), name="hermes-cloud-recv")

    async def _receive(self) -> None:
        error = HermesCloudError("Hermes Cloud connection closed")
        try:
            async for raw in self.websocket:
                for frame in str(raw).splitlines():
                    if not frame:
                        continue
                    try:
                        message = json.loads(frame)
                    except json.JSONDecodeError:
                        continue
                    request_id = message.get("id")
                    if request_id in self._pending:
                        future = self._pending.pop(request_id)
                        error = message.get("error")
                        if error:
                            future.set_exception(
                                HermesCloudError(str(error.get("message") or "Hermes RPC failed"))
                            )
                        else:
                            future.set_result(message.get("result"))
                    if message.get("method") == "event":
                        self._events.put_nowait(message.get("params") or {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = HermesCloudError(f"Hermes Cloud connection closed: {exc}")
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()

    async def request(self, method: str, params: Optional[dict] = None, timeout: float = 1800) -> Any:
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self.websocket.send(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        )
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)

    async def wait_event(self, predicate: Callable[[dict], bool], timeout: float) -> dict:
        async def _wait() -> dict:
            while True:
                event = await self._events.get()
                if predicate(event):
                    return event

        return await asyncio.wait_for(_wait(), timeout=timeout)

    async def close(self) -> None:
        self._receiver.cancel()
        try:
            await self.websocket.close()
        finally:
            try:
                await self._receiver
            except (asyncio.CancelledError, Exception):
                pass


class HermesCloudClient:
    """Send prompts to one existing Nous-hosted Hermes gateway."""

    def __init__(
        self,
        gateway_url: str,
        *,
        token_file: Optional[str] = None,
        timeout: float = 1800,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        if not self.gateway_url.startswith("https://"):
            raise ValueError("HERMES_CLOUD_GATEWAY_URL must use HTTPS")
        configured_file = token_file or os.getenv("HERMES_CLOUD_TOKEN_FILE", "")
        self.token_file = Path(configured_file).expanduser() if configured_file else Path(
            "~/.config/hermes-cloud-bridge/tokens.json"
        ).expanduser()
        self.timeout = max(30.0, float(timeout))
        self._tokens: Optional[Dict[str, Any]] = None
        self._token_lock = asyncio.Lock()
        self._sessions: Dict[str, str] = {}
        self._conversation_locks: Dict[str, asyncio.Lock] = {}

    def _load_tokens(self) -> Dict[str, Any]:
        if self._tokens is not None:
            return dict(self._tokens)
        try:
            tokens = json.loads(self.token_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            tokens = {}
        tokens.setdefault("accessToken", os.getenv("HERMES_CLOUD_ACCESS_TOKEN", ""))
        tokens.setdefault("refreshToken", os.getenv("HERMES_CLOUD_REFRESH_TOKEN", ""))
        tokens.setdefault("expiresAt", os.getenv("HERMES_CLOUD_TOKEN_EXPIRES_AT", "0"))
        tokens.setdefault("provider", os.getenv("HERMES_CLOUD_AUTH_PROVIDER", "nous"))
        self._tokens = tokens
        return dict(tokens)

    def _save_tokens(self, tokens: Dict[str, Any]) -> None:
        self._tokens = dict(tokens)
        try:
            self.token_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.token_file.write_text(json.dumps(tokens, indent=2) + "\n", encoding="utf-8")
            self.token_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            # A read-only secret mount is valid; the refreshed token remains in
            # memory for this process. Operators should use a persistent volume
            # when refresh-token rotation must survive restarts.
            pass

    async def _json_request(self, path: str, *, token: str = "", body: Optional[dict] = None) -> dict:
        headers = {"content-type": "application/json"}
        if token:
            headers["authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "POST" if body is not None else "GET",
                f"{self.gateway_url}{path}",
                headers=headers,
                json=body,
            )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.is_success:
            detail = payload.get("detail") or payload.get("error") or response.reason_phrase
            raise HermesCloudError(f"Hermes Cloud returned {response.status_code}: {detail}")
        return payload

    async def _access_token(self) -> str:
        async with self._token_lock:
            tokens = self._load_tokens()
            access_token = _token_value(tokens, "accessToken", "access_token")
            try:
                expires_at = float(tokens.get("expiresAt") or tokens.get("expires_at") or 0)
            except (TypeError, ValueError):
                expires_at = 0
            if access_token and expires_at > time.time() + 60:
                return access_token

            refresh_token = _token_value(tokens, "refreshToken", "refresh_token")
            if not refresh_token:
                raise HermesCloudError(
                    "Hermes Cloud authentication is missing; provide HERMES_CLOUD_TOKEN_FILE "
                    "or HERMES_CLOUD_REFRESH_TOKEN"
                )
            refreshed = await self._json_request(
                "/auth/native/refresh",
                body={
                    "refresh_token": refresh_token,
                    "provider": str(tokens.get("provider") or "nous"),
                },
            )
            access_token = str(refreshed.get("access_token") or "")
            if not access_token:
                raise HermesCloudError("Hermes Cloud returned no access token")
            updated = {
                "accessToken": access_token,
                "refreshToken": str(refreshed.get("refresh_token") or refresh_token),
                "expiresAt": float(refreshed.get("expires_at") or 0),
                "provider": str(refreshed.get("provider") or tokens.get("provider") or "nous"),
                "userId": str(refreshed.get("user_id") or tokens.get("userId") or ""),
            }
            self._save_tokens(updated)
            return access_token

    async def _connect(self) -> _GatewayConnection:
        ticket_payload = await self._json_request(
            "/api/auth/ws-ticket", token=await self._access_token(), body={}
        )
        ticket = str(ticket_payload.get("ticket") or "")
        if not ticket:
            raise HermesCloudError("Hermes Cloud returned no WebSocket ticket")
        from websockets.asyncio.client import connect

        ws_url = self.gateway_url.replace("https://", "wss://", 1)
        websocket = await connect(f"{ws_url}/api/ws?ticket={quote(ticket)}", max_size=16 * 1024 * 1024)
        connection = _GatewayConnection(websocket)
        await connection.wait_event(lambda event: event.get("type") == "gateway.ready", 15.0)
        return connection

    async def ask(self, message: str, *, conversation: str) -> str:
        """Submit one turn and return Hermes' completed text response."""
        message = str(message).strip()
        if not message:
            raise ValueError("message is required")
        key = str(conversation or "discord-voice")
        lock = self._conversation_locks.setdefault(key, asyncio.Lock())
        async with lock:
            for attempt in range(2):
                connection = await self._connect()
                try:
                    session_id = self._sessions.get(key)
                    if not session_id:
                        created = await connection.request(
                            "session.create",
                            {"title": f"Discord Voice: {key}", "source": "discord-voice"},
                            timeout=60.0,
                        )
                        session_id = str((created or {}).get("session_id") or "")
                        if not session_id:
                            raise HermesCloudError("Hermes Cloud returned no session id")
                        self._sessions[key] = session_id

                    completion = asyncio.create_task(
                        connection.wait_event(
                            lambda event: event.get("type") == "message.complete"
                            and event.get("session_id") == session_id,
                            self.timeout,
                        )
                    )
                    try:
                        await connection.request(
                            "prompt.submit",
                            {"session_id": session_id, "text": message},
                            timeout=self.timeout,
                        )
                        event = await completion
                    except Exception as exc:
                        completion.cancel()
                        try:
                            await completion
                        except (asyncio.CancelledError, Exception):
                            pass
                        if attempt == 0 and "not found" in str(exc).lower():
                            self._sessions.pop(key, None)
                            continue
                        raise
                    payload = event.get("payload") or {}
                    return str(
                        payload.get("text")
                        or payload.get("message")
                        or "Hermes completed without text."
                    )
                finally:
                    await connection.close()

            raise HermesCloudError("Hermes Cloud session could not be restored")
