import os
import json
import time
import base64
import audioop
import aiohttp
import asyncio
import websockets
import urllib.parse
import subprocess
from fastapi import FastAPI, WebSocket, APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
router = APIRouter()

# Audio Config
IN_RATE = 8000
OUT_RATE = 8000
MULAW_SILENCE = b'\xff'

# API Keys
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
EL_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE = os.getenv("ELEVENLABS_VOICE_ID")
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Helpers
def ulaw_b64_to_pcm16(b64_data: str) -> bytes:
    return audioop.ulaw2lin(base64.b64decode(b64_data), 2)

def split_frames(pcm: bytes, frame_bytes: int):
    for i in range(0, len(pcm), frame_bytes):
        yield pcm[i : i + frame_bytes]

def ensure_even_bytes(b: bytes) -> bytes:
    return b if len(b) % 2 == 0 else b + b'\x00'

# ------------------------
# Transport Layer
# ------------------------
class TwilioTransport:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stream_sid = None

    async def send_media_payload(self, payload_b64: str):
        try:
            await self.ws.send_text(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload_b64}
            }))
        except: pass

    async def clear_buffer(self):
        try:
            await self.ws.send_text(json.dumps({
                "event": "clear",
                "streamSid": self.stream_sid
            }))
        except: pass

# ------------------------
# Deepgram Streaming STT
# ------------------------
class DeepgramStreamingSTT:
    def __init__(self, on_utterance_complete):
        self.on_utterance_complete = on_utterance_complete
        self.ws = None
        self.connected = False
        self.keep_alive_task = None
        self.current_utterance = []
        self.utterance_start_time = None
        self.transcript_ready_time = None

    async def connect(self):
        params = {
            "model": "nova-2", "encoding": "linear16", "sample_rate": 8000,
            "punctuate": "true", "endpointing": "300", "interim_results": "true",
        }
        url = "wss://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)
        self.ws = await websockets.connect(url, extra_headers={"Authorization": f"Token {DG_KEY}"})
        self.connected = True
        self.keep_alive_task = asyncio.create_task(self._keep_alive())
        asyncio.create_task(self._recv_loop())

    async def _keep_alive(self):
        while self.connected:
            await asyncio.sleep(5)
            if self.ws: await self.ws.send(json.dumps({"type": "KeepAlive"}))

    async def _recv_loop(self):
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("type") != "Results": continue
            transcript = data["channel"]["alternatives"][0].get("transcript", "").strip()
            if not transcript: 
                continue

            if data.get("is_final"):
                self.current_utterance.append(transcript)

            if data.get("speech_final"):
                self.transcript_ready_time = time.time()
                full_text = " ".join(self.current_utterance).strip()
                self.current_utterance.clear()

                if full_text:
                    # Compute STT metrics
                    if self.utterance_start_time:
                        ttft = self.transcript_ready_time - self.utterance_start_time
                        ttfb = self.transcript_ready_time - self.utterance_start_time  # For Deepgram, TTFT ~ TTFB
                        print(f"[STT] TTFB: {ttfb:.3f}s, TTFT: {ttft:.3f}s")
                        self.utterance_start_time = None  # reset
                    await self.on_utterance_complete(full_text)

    async def send_audio(self, pcm16: bytes):
        if self.connected:
            # mark first audio frame for metrics
            if not self.utterance_start_time:
                self.utterance_start_time = time.time()
            await self.ws.send(pcm16)

    async def close(self):
        self.connected = False
        if self.keep_alive_task: self.keep_alive_task.cancel()
        if self.ws: await self.ws.close()

# ------------------------
# Custom LLM (File Prompt)
# ------------------------
class CustomLLM:
    def __init__(self, prompt_file_name="prompt.txt"):
        self.client = AsyncOpenAI(api_key="any", base_url="https://llm.dev.voicing.ai/v1")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(current_dir, prompt_file_name)
        with open(prompt_file_path, "r") as f:
            self.system_prompt = f.read()
        self.system_prompt_added = False

    async def run(self, conversation_history: list) -> str:
        messages = []
        if not self.system_prompt_added:
            messages.append({"role": "system", "content": self.system_prompt})
            self.system_prompt_added = True

        messages += conversation_history[-10:]
        start_time = time.time()
        response = await self.client.chat.completions.create(
            model="voicing-llm-v1.5",
            messages=messages,
            stream=True
        )
        full_reply = []
        async for chunk in response:
            if chunk.choices[0].delta.content:
                full_reply.append(chunk.choices[0].delta.content)
        ttfb = time.time() - start_time
        print(f"[LLM] TTFB: {ttfb:.3f}s")
        return "".join(full_reply).strip()

# ------------------------
# ElevenLabs TTS
# ------------------------
class ElevenLabsTTS:
    async def run(self, text: str) -> bytes:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ELEVEN_URL}/{EL_VOICE}",
                json={"text": text, "model_id": "eleven_multilingual_v2", "output_format": "mp3_44100_128"},
                headers={"xi-api-key": EL_KEY}
            ) as resp:
                raw = await resp.read()
                ff = subprocess.run(
                    ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "8000", "pipe:1"],
                    input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                audio = audioop.lin2ulaw(ensure_even_bytes(ff.stdout), 2)
        ttfb = time.time() - start_time
        print(f"[TTS] TTFB: {ttfb:.3f}s")
        return audio

# ------------------------
# Pipeline & Orchestration
# ------------------------
class VoicePipeline:
    def __init__(self, llm, tts, transport):
        self.llm = llm
        self.tts = tts
        self.transport = transport
        self.playback_queue = asyncio.Queue()
        self.is_playing_tts = False
        self.current_tts_task = None
        self.stt = DeepgramStreamingSTT(self._on_stt_utterance)
        self.history = []
        self.first_interaction_done = False
        asyncio.create_task(self.stt.connect())
        asyncio.create_task(self._playback_worker())

    async def _on_stt_utterance(self, text):
        user_received_time = time.time()
        print(f"[STT] User said: {text}")
        self.history.append({"role": "user", "content": text})

        # Non-blocking greeting for first interaction
        if not self.first_interaction_done:
            self.first_interaction_done = True
            greeting = "Hello, this is Alex from IT Support. How can I help you today?"
            self.history.append({"role": "assistant", "content": greeting})
            asyncio.create_task(self.playback_queue.put(await self.tts.run(greeting)))

        # LLM processing
        reply = await self.llm.run(self.history)
        self.history.append({"role": "assistant", "content": reply})

        # TTS processing
        audio = await self.tts.run(reply)

        # TTFT metric
        ttft = time.time() - user_received_time
        print(f"[Voice Pipeline] TTFT: {ttft:.3f}s")

        await self.playback_queue.put(audio)

    async def _playback_worker(self):
        while True:
            audio = await self.playback_queue.get()
            self.current_tts_task = asyncio.create_task(self._play_audio(audio))
            try:
                await self.current_tts_task
            except asyncio.CancelledError:
                pass
            finally:
                self.is_playing_tts = False
                self.current_tts_task = None
                self.playback_queue.task_done()

    async def _play_audio(self, audio):
        self.is_playing_tts = True
        for frame in split_frames(audio, 160):
            await self.transport.send_media_payload(base64.b64encode(frame).decode())
            await asyncio.sleep(0.019)

    async def accept_audio(self, pcm16):
        await self.stt.send_audio(pcm16)
        if self.is_playing_tts and audioop.rms(pcm16, 2) > 600:  # Barge-in
            if self.current_tts_task:
                self.current_tts_task.cancel()
            await self.transport.clear_buffer()
            while not self.playback_queue.empty():
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()

# ------------------------
# Routes
# ------------------------
@app.post("/incoming_call")
async def incoming_call(request: Request):
    base_url = os.getenv("PUBLIC_WEBSOCKET_URL", "").rstrip("/")
    stream_url = f"{base_url}/twilio-stream"
    resp = VoiceResponse()
    connect = Connect()
    connect.append(Stream(url=stream_url))
    resp.append(connect)
    return Response(content=str(resp), media_type="application/xml")

@router.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    transport = TwilioTransport(ws)
    pipeline = VoicePipeline(
        llm=CustomLLM("prompt.txt"), 
        tts=ElevenLabsTTS(), 
        transport=transport
    )

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg["event"] == "start":
                transport.stream_sid = msg["start"]["streamSid"]
            elif msg["event"] == "media":
                pcm_in = ulaw_b64_to_pcm16(msg["media"]["payload"])
                await pipeline.accept_audio(pcm_in)
            elif msg["event"] == "stop":
                await pipeline.stt.close()
                break
    except Exception as e:
        print(f"WS Error: {e}")
    finally:
        print("WebSocket closed")

app.include_router(router)