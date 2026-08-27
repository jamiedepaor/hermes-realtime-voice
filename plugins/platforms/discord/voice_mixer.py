from __future__ import annotations

"""
Continuous PCM audio mixer for Discord voice channels.

discord.py (Rapptz) ships no audio mixer: ``VoiceClient.play()`` accepts a
single :class:`discord.AudioSource` and raises ``ClientException`` if called
while already playing.  One opus stream per connection, one source feeding it.

This module adds software mixing *upstream* of that single stream.  A
:class:`VoiceMixer` is itself a ``discord.AudioSource`` that discord.py polls
every 20 ms via :meth:`read`.  Internally it sums the 20 ms PCM frames of any
number of child sources, clamps to int16, and returns one blended frame.
discord.py never knows several streams were combined underneath — it just
encodes and sends the single mixed frame.

This gives us, for one voice connection at once:

  * an always-on low-volume **ambient/idle loop** (the "thinking" sound),
  * a **speech** channel (TTS replies, verbal acknowledgements) that plays
    *over* the ambient bed, automatically **ducking** the ambient gain down
    while speech is active and restoring it when speech ends — the smooth
    Grok-voice-mode feel, instead of stop-and-swap.

Design notes
------------
* The mixer is installed **once** per guild on join (``vc.play(mixer)``) and
  runs continuously until the bot leaves.  Children come and go; the mixer
  itself never stops, so there is no ``is_playing()`` race between an
  acknowledgement and the final reply.
* Frame format is Discord-native: 48 kHz, 2 channels, signed 16-bit LE,
  20 ms per frame == ``discord.opus.Encoder.FRAME_SIZE`` bytes
  (3840 = 960 samples * 2 channels * 2 bytes).
* Mixing is a single vectorised int32 add + clip per 20 ms frame (numpy,
  already a core dependency).  CPU cost is negligible.
* :meth:`read` is called from discord.py's audio sender **thread**, while
  children are added/removed from the asyncio event loop thread, so all
  shared state is guarded by a plain ``threading.Lock``.

The mixer NEVER touches the inbound receive path: it only produces the bot's
*outgoing* stream.  The :class:`VoiceReceiver` decodes incoming SSRCs only, so
the mixer's output cannot echo back into transcription.
"""

import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import discord

try:
    from .ffmpeg_utils import resolve_ffmpeg_executable
except ImportError:
    from ffmpeg_utils import resolve_ffmpeg_executable

if TYPE_CHECKING:  # numpy is an optional ("voice" extra) dep — never import at runtime top-level
    import numpy as np

logger = logging.getLogger(__name__)


def _require_numpy():
    """Import numpy lazily.

    numpy ships in the optional ``voice`` extra, not the base install, so this
    module must import cleanly without it (the Discord adapter imports this
    file unconditionally).  Callers that actually mix audio call this; if the
    voice extra isn't installed they get a clear error instead of a top-level
    ImportError that would break the whole adapter import.
    """
    import numpy as np  # noqa: PLC0415 — intentional lazy import
    return np

# Discord-native frame geometry (matches discord.opus.Encoder).
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2                       # bytes per sample (s16)
FRAME_LENGTH_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_LENGTH_MS // 1000   # 960
FRAME_SIZE = SAMPLES_PER_FRAME * CHANNELS * SAMPLE_WIDTH    # 3840 bytes
BYTES_PER_MS = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH // 1000  # 192
SILENCE_FRAME = b"\x00" * FRAME_SIZE


class MixerChild:
    """A single audio stream feeding into :class:`VoiceMixer`.

    Wraps raw 48 kHz / stereo / s16le PCM bytes.  ``read_frame`` hands back one
    20 ms frame at a time, optionally looping, with a per-child gain applied.
    """

    __slots__ = (
        "name", "_pcm", "_pos", "loop", "gain",
        "is_speech", "fade_frames", "_fade_done", "_finished",
    )

    def __init__(
        self,
        name: str,
        pcm: bytes,
        *,
        loop: bool = False,
        gain: float = 1.0,
        is_speech: bool = False,
        fade_in_ms: int = 0,
    ):
        # Pad to a whole number of frames so looping is seamless and the final
        # partial frame doesn't click.
        remainder = len(pcm) % FRAME_SIZE
        if remainder:
            pcm = pcm + b"\x00" * (FRAME_SIZE - remainder)
        self.name = name
        self._pcm = pcm
        self._pos = 0
        self.loop = loop
        self.gain = float(gain)
        self.is_speech = is_speech
        # Linear fade-in over N frames avoids a click when a loud child starts.
        self.fade_frames = max(0, fade_in_ms // FRAME_LENGTH_MS)
        self._fade_done = 0
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def read_frame(self) -> "Optional[np.ndarray]":
        """Return the next 20 ms frame as an int16 ndarray, or None if done."""
        if self._finished:
            return None
        if self._pos >= len(self._pcm):
            if self.loop and self._pcm:
                self._pos = 0
            else:
                self._finished = True
                return None

        np = _require_numpy()
        chunk = self._pcm[self._pos:self._pos + FRAME_SIZE]
        self._pos += FRAME_SIZE
        if len(chunk) < FRAME_SIZE:
            chunk = chunk + b"\x00" * (FRAME_SIZE - len(chunk))

        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)

        gain = self.gain
        if self.fade_frames and self._fade_done < self.fade_frames:
            self._fade_done += 1
            gain *= self._fade_done / self.fade_frames

        if gain != 1.0:
            samples = samples * gain
        return samples


class StreamingMixerChild:
    """A speech child that accepts PCM incrementally.

    OpenAI Realtime emits small audio deltas while a response is being
    generated.  Buffering those deltas into a complete file would throw away
    the latency benefit, while adding each delta as a normal ``MixerChild``
    would mix them concurrently.  This child keeps one ordered byte stream
    that the Discord sender drains frame-by-frame.

    All methods are called while the owning ``VoiceMixer`` lock is held.
    """

    __slots__ = (
        "name", "stream_id", "gain", "is_speech", "fade_frames",
        "_fade_done", "_buffer", "_pos", "_ended", "_finished",
        "_played_bytes",
    )

    def __init__(
        self,
        stream_id: str,
        *,
        gain: float = 1.0,
        fade_in_ms: int = 40,
    ) -> None:
        self.name = f"stream:{stream_id}"
        self.stream_id = stream_id
        self.gain = float(gain)
        self.is_speech = True
        self.fade_frames = max(0, fade_in_ms // FRAME_LENGTH_MS)
        self._fade_done = 0
        self._buffer = bytearray()
        self._pos = 0
        self._ended = False
        self._finished = False
        self._played_bytes = 0

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def played_ms(self) -> int:
        return self._played_bytes // BYTES_PER_MS

    def append(self, pcm: bytes) -> None:
        if pcm and not self._ended and not self._finished:
            self._buffer.extend(pcm)

    def finish(self) -> None:
        self._ended = True

    def cancel(self) -> int:
        played_ms = self.played_ms
        self._buffer.clear()
        self._pos = 0
        self._ended = True
        self._finished = True
        return played_ms

    def read_frame(self) -> "Optional[np.ndarray]":
        if self._finished:
            return None

        available = len(self._buffer) - self._pos
        if available <= 0:
            if self._ended:
                self._finished = True
                return None
            # The provider can briefly produce slower than Discord consumes.
            # Keep the stream alive and preserve ordering instead of declaring
            # it finished and dropping later deltas.
            np = _require_numpy()
            return np.zeros(FRAME_SIZE // SAMPLE_WIDTH, dtype=np.float32)

        take = min(FRAME_SIZE, available)
        chunk = bytes(self._buffer[self._pos:self._pos + take])
        self._pos += take
        self._played_bytes += take

        if take < FRAME_SIZE:
            if not self._ended:
                # Wait for the rest of this frame. Roll back the read so bytes
                # remain ordered when the next provider delta arrives.
                self._pos -= take
                self._played_bytes -= take
                np = _require_numpy()
                return np.zeros(FRAME_SIZE // SAMPLE_WIDTH, dtype=np.float32)
            chunk += b"\x00" * (FRAME_SIZE - take)

        # Periodically discard committed bytes so a long spoken response does
        # not retain its complete PCM history in memory.  Do this only after a
        # complete (or final padded) frame; partial reads may be rolled back.
        if self._pos >= 1024 * 1024:
            del self._buffer[:self._pos]
            self._pos = 0

        np = _require_numpy()
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        gain = self.gain
        if self.fade_frames and self._fade_done < self.fade_frames:
            self._fade_done += 1
            gain *= self._fade_done / self.fade_frames
        if gain != 1.0:
            samples = samples * gain
        return samples


class VoiceMixer(discord.AudioSource):
    """A continuous ``discord.AudioSource`` that mixes N child streams.

    Use :meth:`set_ambient` to install/replace the looping idle bed and
    :meth:`play_speech` to layer a one-shot clip over it (ducking the ambient
    while it plays).  Both are safe to call from the asyncio loop thread while
    discord.py drains :meth:`read` from its sender thread.
    """

    # discord.AudioSource subclasses set is_opus()==False to receive PCM.
    def is_opus(self) -> bool:  # pragma: no cover - trivial
        return False

    def __init__(
        self,
        *,
        ambient_gain: float = 0.18,
        duck_gain: float = 0.06,
        speech_gain: float = 1.0,
        duck_release_ms: int = 400,
    ):
        self._lock = threading.Lock()
        self._ambient: Optional[MixerChild] = None
        self._speech: List[Union[MixerChild, StreamingMixerChild]] = []
        self._speech_streams: Dict[str, StreamingMixerChild] = {}
        self._ambient_gain = float(ambient_gain)
        self._duck_gain = float(duck_gain)
        self._speech_gain = float(speech_gain)
        # When speech ends, ramp the ambient back up over this many frames
        # instead of jumping, so the bed swells back smoothly.
        self._duck_release_frames = max(1, duck_release_ms // FRAME_LENGTH_MS)
        self._duck_release_left = 0
        self._closed = False
        # Tracks whether speech is currently active, for external callers that
        # want to avoid double-ducking or know when a reply is mid-flight.
        self._speech_active = False

    # ------------------------------------------------------------------
    # Ambient (idle / "thinking") bed
    # ------------------------------------------------------------------

    def set_ambient(self, pcm: Optional[bytes], *, gain: Optional[float] = None) -> None:
        """Install (or clear, with ``pcm=None``) the looping ambient bed."""
        with self._lock:
            if gain is not None:
                self._ambient_gain = float(gain)
            if not pcm:
                self._ambient = None
                return
            self._ambient = MixerChild(
                "ambient", pcm, loop=True,
                gain=self._effective_ambient_gain(), fade_in_ms=200,
            )

    def _effective_ambient_gain(self) -> float:
        return self._duck_gain if self._speech_active else self._ambient_gain

    # ------------------------------------------------------------------
    # Speech (TTS replies, verbal acks) layered over the ambient bed
    # ------------------------------------------------------------------

    def play_speech(self, pcm: bytes, *, gain: Optional[float] = None,
                    fade_in_ms: int = 40) -> None:
        """Layer a one-shot speech clip over the ambient bed (ducks ambient)."""
        if not pcm:
            return
        with self._lock:
            child = MixerChild(
                "speech", pcm, loop=False,
                gain=self._speech_gain if gain is None else float(gain),
                is_speech=True, fade_in_ms=fade_in_ms,
            )
            self._speech.append(child)
            self._speech_active = True
            self._duck_release_left = 0
            if self._ambient is not None:
                self._ambient.gain = self._duck_gain

    def start_speech_stream(
        self,
        stream_id: str,
        *,
        gain: Optional[float] = None,
        fade_in_ms: int = 40,
    ) -> None:
        """Start one ordered, incrementally-fed speech stream."""
        if not stream_id:
            return
        with self._lock:
            old = self._speech_streams.pop(stream_id, None)
            if old is not None:
                old.cancel()
                self._speech = [child for child in self._speech if child is not old]
            child = StreamingMixerChild(
                stream_id,
                gain=self._speech_gain if gain is None else float(gain),
                fade_in_ms=fade_in_ms,
            )
            self._speech_streams[stream_id] = child
            self._speech.append(child)
            self._speech_active = True
            self._duck_release_left = 0
            if self._ambient is not None:
                self._ambient.gain = self._duck_gain

    def append_speech_stream(self, stream_id: str, pcm: bytes) -> bool:
        """Append PCM to an existing stream; return False if it is unknown."""
        if not pcm:
            return True
        with self._lock:
            child = self._speech_streams.get(stream_id)
            if child is None:
                return False
            child.append(pcm)
            return True

    def finish_speech_stream(self, stream_id: str) -> bool:
        """Mark a stream complete; buffered frames continue to drain."""
        with self._lock:
            child = self._speech_streams.get(stream_id)
            if child is None:
                return False
            child.finish()
            return True

    def cancel_speech_stream(self, stream_id: str) -> int:
        """Drop an in-flight stream and return audio milliseconds consumed."""
        with self._lock:
            child = self._speech_streams.pop(stream_id, None)
            if child is None:
                return 0
            played_ms = child.cancel()
            self._speech = [item for item in self._speech if item is not child]
            if not self._speech:
                self._begin_duck_release_locked()
            return played_ms

    @property
    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_active

    def stop_speech(self) -> None:
        """Drop any in-flight speech immediately and release the duck."""
        with self._lock:
            for child in self._speech_streams.values():
                child.cancel()
            self._speech_streams.clear()
            self._speech.clear()
            self._begin_duck_release_locked()

    def _begin_duck_release_locked(self) -> None:
        self._speech_active = False
        self._duck_release_left = self._duck_release_frames

    # ------------------------------------------------------------------
    # AudioSource interface — called from discord.py's sender thread
    # ------------------------------------------------------------------

    def read(self) -> bytes:
        """Return one 20 ms mixed PCM frame (always FRAME_SIZE bytes).

        Returning a non-empty frame keeps discord.py's player alive; we never
        return b"" because that would stop the single underlying stream and we
        want the mixer to run continuously for the lifetime of the connection.
        """
        with self._lock:
            if self._closed:
                return SILENCE_FRAME

            np = _require_numpy()
            acc: "Optional[np.ndarray]" = None

            # Speech children (drop exhausted ones; release duck when last ends)
            if self._speech:
                still_live: List[MixerChild] = []
                for child in self._speech:
                    frame = child.read_frame()
                    if frame is None:
                        if isinstance(child, StreamingMixerChild):
                            self._speech_streams.pop(child.stream_id, None)
                        continue
                    acc = frame if acc is None else acc + frame
                    still_live.append(child)
                self._speech = still_live
                if not self._speech and self._speech_active:
                    self._begin_duck_release_locked()

            # Ambient bed — ramp gain back up during duck-release.
            if self._ambient is not None:
                if self._duck_release_left > 0 and not self._speech_active:
                    self._duck_release_left -= 1
                    frac = 1.0 - (self._duck_release_left / self._duck_release_frames)
                    self._ambient.gain = (
                        self._duck_gain
                        + (self._ambient_gain - self._duck_gain) * frac
                    )
                elif not self._speech_active and self._duck_release_left == 0:
                    self._ambient.gain = self._ambient_gain
                amb = self._ambient.read_frame()
                if amb is not None:
                    acc = amb if acc is None else acc + amb

            if acc is None:
                return SILENCE_FRAME

            np.clip(acc, -32768, 32767, out=acc)
            return acc.astype(np.int16).tobytes()

    def cleanup(self) -> None:  # called by discord.py when playback stops
        with self._lock:
            self._closed = True
            self._ambient = None
            for child in self._speech_streams.values():
                child.cancel()
            self._speech_streams.clear()
            self._speech.clear()


# ----------------------------------------------------------------------
# PCM helpers
# ----------------------------------------------------------------------

def decode_to_pcm(path: str, *, timeout: float = 30.0) -> Optional[bytes]:
    """Decode any audio file to 48 kHz / stereo / s16le PCM via ffmpeg.

    Returns the raw PCM bytes, or None on failure.  ffmpeg is already a hard
    requirement of the voice path (see ``VoiceReceiver.pcm_to_wav``).
    """
    import subprocess

    try:
        proc = subprocess.run(
            [
                resolve_ffmpeg_executable(), "-y", "-loglevel", "error",
                "-i", path,
                "-f", "s16le",
                "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS),
                "pipe:1",
            ],
            capture_output=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("decode_to_pcm failed for %s: %s", path, e)
        return None
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg decode failed for %s (rc=%d): %s",
            path, proc.returncode, (proc.stderr or b"").decode("utf-8", "replace")[:200],
        )
        return None
    return proc.stdout or None


def synth_ambient_pcm(seconds: float = 4.0) -> bytes:
    """Synthesise a subtle looping ambient bed (no asset file required).

    A soft, slowly-pulsing low pad: two detuned sine partials with a gentle
    tremolo, plus a touch of filtered noise.  Designed to loop seamlessly
    (whole number of cycles, zero-crossing endpoints) and sit quietly under
    speech.  Mono content duplicated to stereo.
    """
    np = _require_numpy()
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    # Choose base frequencies that complete whole cycles over the loop so the
    # wrap point is click-free.
    def _whole_cycle_freq(target: float) -> float:
        cycles = max(1, round(target * seconds))
        return cycles / seconds

    f1 = _whole_cycle_freq(110.0)
    f2 = _whole_cycle_freq(110.5)
    trem = _whole_cycle_freq(0.5)   # ~0.5 Hz tremolo

    pad = (
        0.55 * np.sin(2 * np.pi * f1 * t)
        + 0.45 * np.sin(2 * np.pi * f2 * t)
    )
    tremolo = 0.6 + 0.4 * (0.5 * (1 + np.sin(2 * np.pi * trem * t)))
    signal = pad * tremolo

    # Smooth filtered noise for air, kept very low.
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    kernel = np.ones(64) / 64.0
    noise = np.convolve(noise, kernel, mode="same")
    signal = signal + 0.08 * noise

    # Normalise to a modest peak (mixer applies the real ambient gain on top).
    peak = float(np.max(np.abs(signal))) or 1.0
    signal = (signal / peak) * 0.5

    mono16 = (signal * 32767.0).astype(np.int16)
    stereo16 = np.repeat(mono16[:, None], CHANNELS, axis=1).reshape(-1)
    return stereo16.tobytes()
