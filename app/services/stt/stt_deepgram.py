import asyncio
from typing import Optional, Callable
from deepgram import DeepgramClient, LiveTranscriptionEvents

class DeepgramRealtimeSTT:
    """
    Lightweight Deepgram real-time STT module.
    Accepts PCM16 audio (8000 Hz) and emits transcription text.

    Usage:
        stt = DeepgramRealtimeSTT(api_key)
        await stt.start()
        await stt.send_audio(pcm16_bytes)
        text = await stt.get_transcript()   # returns interim or final text
    """

    def __init__(self, api_key: str, sample_rate: int = 8000):
        self.api_key = api_key
        self.sample_rate = sample_rate

        self._client = DeepgramClient(api_key)
        self._socket = None

        # Queue where transcripts are pushed
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def start(self):
        """Start Deepgram WS + register callbacks."""
        self._socket = self._client.listen.live.v("1")

        # --- Register transcript callback ---
        self._socket.on(
            LiveTranscriptionEvents.Transcript,
            self._on_transcript
        )

        options = LiveTranscriptionOptions(
            encoding="linear16",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=True,
            punctuate=True,
            smart_format=True,
            model="nova-2"
        )

        await self._socket.start(options)

    async def stop(self):
        if self._socket:
            await self._socket.finish()

    # -----------------------
    # CALLBACK FROM DEEPGRAM
    # -----------------------
    async def _on_transcript(self, result, **kwargs):
        if len(result.channel.alternatives) == 0:
            return

        transcript = result.channel.alternatives[0].transcript
        if transcript.strip():
            await self._queue.put(transcript)

    # -----------------------
    # SEND AUDIO TO DEEPGRAM
    # -----------------------
    async def send_audio(self, pcm16: bytes):
        if self._socket:
            await self._socket.send(pcm16)

    # -----------------------
    # READ TRANSCRIPTION
    # -----------------------
    async def get_transcript(self) -> Optional[str]:
        """
        Returns next available transcript (interim or final).
        If none available, returns None.
        """
        if self._queue.empty():
            return None
        return await self._queue.get()
