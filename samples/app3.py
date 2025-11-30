import os
import json
import base64
import audioop
import aiohttp
import asyncio
import io
import wave
from fastapi import FastAPI, WebSocket, APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from openai import AsyncOpenAI
from dotenv import load_dotenv



load_dotenv()
IN_RATE = 8000
OUT_RATE = 8000
MULAW_SILENCE = b'\xff'

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
DG_KEY = os.getenv("DEEPGRAM_API_KEY")

ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech"
EL_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE = os.getenv("ELEVENLABS_VOICE_ID")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_KEY)

def ulaw_b64_to_pcm16(b64_data: str) -> bytes:
    ulaw_bytes = base64.b64decode(b64_data)
    return audioop.ulaw2lin(ulaw_bytes, 2)

def pad_frame(frame: bytes, frame_size: int = 160) -> bytes:
    if len(frame) == frame_size:
        return frame
    if len(frame) > frame_size:
        return frame[:frame_size]
    # pad with µ-law silence byte
    return frame + (MULAW_SILENCE * (frame_size - len(frame)))

def mulaw_rms(frame: bytes) -> float:
    # convert small mulaw chunk -> pcm16 then compute RMS
    try:
        pcm = audioop.ulaw2lin(frame, 2)
        return audioop.rms(pcm, 2)
    except Exception:
        return 0.0

# def pcm16_to_ulaw_b64(pcm16: bytes) -> str:
#     ulaw = audioop.lin2ulaw(pcm16, 2)
#     return base64.b64encode(ulaw).decode()

def split_frames(pcm: bytes, frame_bytes: int):
    """Yield sequential frame-sized byte chunks"""
    for i in range(0, len(pcm), frame_bytes):
        yield pcm[i : i + frame_bytes]

def ensure_even_bytes(b: bytes) -> bytes:
    return b if len(b) % 2 == 0 else b + b'\x00'


# ======================================================
# Pipecat-Inspired Custom Architecture
# ======================================================

# ------------------------
# Transport Layer
# ------------------------
class TwilioTransport:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stream_sid = None

    async def receive(self):
        msg = await self.ws.receive_text()
        return json.loads(msg)

    async def send_media_payload(self, payload_b64: str):
        # safe send (twilio may close)
        try:
            await self.ws.send_text(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload_b64}
            }))
        except Exception:
            # socket closed or send failed
            pass



# ------------------------
#  REAL STT SERVICE (Deepgram)
# ------------------------
# ------------------------
#  REAL STT SERVICE (Deepgram)
# ------------------------
class DeepgramSTT:
    async def run(self, pcm16: bytes) -> str:
        # send PCM16 chunk to Deepgram REST /listen (synchronous transcription for chunk)
        if not DG_KEY:
            return ""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    DEEPGRAM_URL,
                    data=pcm16,
                    headers={
                        "Content-Type": "audio/pcm",
                        "Authorization": f"Token {DG_KEY}",
                    },
                    params={"model": "nova-2", "encoding": "linear16", "sample_rate": 8000},
                    timeout=10,
                ) as resp:
                    if resp.status != 200:
                        txt = await resp.text()
                        print("Deepgram HTTP Error", resp.status, txt)
                        return ""
                    out = await resp.json()
                    # navigate safely
                    try:
                        return out["results"]["channels"][0]["alternatives"][0].get("transcript","") or ""
                    except Exception:
                        return ""
            except Exception as e:
                print("Deepgram request failed:", e)
                return ""

# ------------------------
# REAL LLM (OpenAI GPT-4o-mini)
# ------------------------
class OpenAILLM:
    async def run(self, text: str) -> str:
        if not OPENAI_KEY:
            return "No OpenAI key configured."
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise phone assistant. Keep responses short."},
                    {"role": "user", "content": text},
                ],
                max_tokens=60,
            )
            # new SDK: message is an object
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print("OpenAI error:", e)
            return ""
# ------------------------
# REAL TTS (ElevenLabs)
# ------------------------

import subprocess
from typing import Optional

# optional import; not required but preferred for MP3 decoding
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False
class ElevenLabsTTS:
    async def run(self, text: str) -> bytes:
        if not EL_KEY or not EL_VOICE:
            print("ElevenLabsTTS: API key or voice ID missing, returning silence")
            return b'\xff' * 160  # 20ms silence (µ-law)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{ELEVEN_URL}/{EL_VOICE}",
                    json={"text": text, "model_id": "eleven_multilingual_v2", "output_format": "wav"},
                    headers={"xi-api-key": EL_KEY},
                    timeout=20,
                ) as resp:

                    status = resp.status
                    content_type = resp.headers.get("Content-Type", "")
                    raw = await resp.read()

                    if status != 200:
                        body = raw.decode(errors="ignore")
                        print(f"TTS HTTP error {status}: {body}")
                        return b'\xff' * 160

            except Exception as e:
                print("ElevenLabs request failed:", e)
                return b'\xff' * 160

        # If the response is already µ-law / mulaw_8000 (rare), return as-is
        if "mulaw" in content_type or "ulaw" in content_type:
            # ensure even bytes
            return ensure_even_bytes(raw)

        # If we were returned WAV-like data (Content-Type contains 'wav' or starts with RIFF bytes)
        if content_type and ("wav" in content_type.lower() or raw[:4] in (b'RIFF', b'RIFX')):
            try:
                with io.BytesIO(raw) as buf:
                    with wave.open(buf, "rb") as wf:
                        frames = wf.readframes(wf.getnframes())
                        sr = wf.getframerate()
                        chans = wf.getnchannels()
                        sw = wf.getsampwidth()

                pcm = frames
                if sw != 2:
                    pcm = audioop.lin2lin(pcm, sw, 2)
                if chans > 1:
                    pcm = audioop.tomono(pcm, 2, 0.5, 0.5)

                if sr != 8000:
                    pcm = audioop.ratecv(pcm, 2, 1, sr, 8000, None)[0]

                mulaw = audioop.lin2ulaw(ensure_even_bytes(pcm), 2)
                mulaw = ensure_even_bytes(mulaw)
                print(f"ElevenLabsTTS: WAV decoded -> {len(mulaw)} bytes (µ-law 8k)")
                return mulaw

            except Exception as e:
                print("WAV decode failed:", e)
                # fallthrough to MP3 handling

        # If response is MP3/MP2/Audio-MPEG, try to decode
        if "mpeg" in content_type or "mp3" in content_type or raw[:3] == b'\xff\xfb':
            # Try pydub first (needs ffmpeg in PATH)
            if PYDUB_AVAILABLE:
                try:
                    seg = AudioSegment.from_file(io.BytesIO(raw), format="mp3")
                    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    pcm = seg.raw_data
                    mulaw = audioop.lin2ulaw(ensure_even_bytes(pcm), 2)
                    mulaw = ensure_even_bytes(mulaw)
                    print(f"ElevenLabsTTS: MP3 decoded via pydub -> {len(mulaw)} bytes (µ-law 8k)")
                    return mulaw
                except Exception as e:
                    print("pydub MP3 decode failed:", e)

            # Fallback: use ffmpeg subprocess to convert mp3 -> wav -> raw pcm
            try:
                # run ffmpeg to convert to PCM16LE mono 8000Hz to stdout
                ff = subprocess.run(
                    ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "8000", "pipe:1"],
                    input=raw,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                pcm = ff.stdout
                mulaw = audioop.lin2ulaw(ensure_even_bytes(pcm), 2)
                mulaw = ensure_even_bytes(mulaw)
                print(f"ElevenLabsTTS: MP3 decoded via ffmpeg -> {len(mulaw)} bytes (µ-law 8k)")
                return mulaw
            except FileNotFoundError:
                print("ffmpeg not found on PATH — install ffmpeg or pip install pydub + ffmpeg")
            except subprocess.CalledProcessError as e:
                print("ffmpeg conversion failed:", e.stderr.decode(errors="ignore")[:200])

        # Unknown format: as a last resort, attempt to treat raw as PCM16@16000 and resample
        try:
            pcm = raw
            # attempt to resample 16k->8k then convert
            pcm = ensure_even_bytes(pcm)
            mulaw = audioop.lin2ulaw(audioop.ratecv(pcm, 2, 1, 16000, 8000, None)[0], 2)
            mulaw = ensure_even_bytes(mulaw)
            print(f"ElevenLabsTTS: fallback raw->mulaw produced {len(mulaw)} bytes")
            return mulaw
        except Exception as e:
            print("TTS fallback conversion failed:", e)

        # ultimate fallback: silence
        return b'\xff' * 160



# ------------------------
# Pipeline (Pipecat Style)
# ------------------------
class VoicePipeline:
    def __init__(self, stt, llm, tts, transport: TwilioTransport):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.transport = transport
        self.buffer = b""
        self.MIN_MS = 1000  # optional, can be used for minimal buffering
        self.BYTES_PER_MS = (IN_RATE * 2) // 1000
        self.outgoing_lock = asyncio.Lock()
        self.processing_tasks = set()
        self.silence_frames = 0  # <-- initialize silence counter

    async def flush_buffer(self):
        """Force process whatever is in buffer (e.g., on call end)."""
        if len(self.buffer) > 0:
            chunk = self.buffer
            self.buffer = b""
            task = asyncio.create_task(self._process_and_play(chunk))
            self.processing_tasks.add(task)
            task.add_done_callback(lambda t: self.processing_tasks.discard(t))

    async def accept_incoming(self, pcm16: bytes):
        """Called for each incoming media event. Non-blocking: spawns background work when buffer full."""
        self.buffer += pcm16

        # --- silence detection ---
        energy = audioop.rms(pcm16, 2)
        if energy < 200:   # tweak this threshold
            self.silence_frames += 1
        else:
            self.silence_frames = 0

        # 10 consecutive silent frames (~200ms) = user finished speaking
        if self.silence_frames > 10 and len(self.buffer) > 0:
            chunk = self.buffer
            self.buffer = b""
            task = asyncio.create_task(self._process_and_play(chunk))
            self.processing_tasks.add(task)
            task.add_done_callback(lambda t: self.processing_tasks.discard(t))

    async def _process_and_play(self, chunk: bytes):
        try:
            text = await self.stt.run(chunk)
            print("STT:", text)
            if not text.strip():
                return

            reply = await self.llm.run(text)
            print("LLM:", reply)
            if not reply:
                return

            # PCM already in mulaw_8000 bytes (no need to convert)
            pcm_ulaw = await self.tts.run(reply)

            print(f"TTS generated {len(pcm_ulaw)} bytes (mulaw_8000)")

            # 20ms @ 8000Hz = 160 bytes
            frame_bytes = 160

            async with self.outgoing_lock:
                for frame in split_frames(pcm_ulaw, frame_bytes):

                    if not frame:
                        continue

                    # SEND DIRECTLY AS BASE64 (NO CONVERSION)
                    payload_b64 = base64.b64encode(frame).decode()

                    await self.transport.send_media_payload(payload_b64)

                    # pacing
                    await asyncio.sleep(0.02)

        except Exception as e:
            print("Pipeline processing error:", e)


# ======================================================
# FastAPI Setup
# ======================================================
app = FastAPI()
router = APIRouter()

@app.post("/incoming_call")
async def incoming_call(request: Request):
    # Twilio should point to this endpoint
    PUBLIC_WSS = os.getenv("PUBLIC_WEBSOCKET_URL", "wss://oss-test.ngrok.app/twilio-stream")
    resp = VoiceResponse()
    connect = Connect()
    connect.append(Stream(url=PUBLIC_WSS))
    resp.append(connect)
    return Response(str(resp), media_type="application/xml")

@router.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    transport = TwilioTransport(ws)
    pipeline = VoicePipeline(
        stt=DeepgramSTT(), llm=OpenAILLM(), tts=ElevenLabsTTS(), transport=transport
    )

    print("Twilio WebSocket connected")
    try:
        while True:
            data = await transport.ws.receive_text()
            msg = json.loads(data)
            event = msg.get("event")

            if event == "start":
                transport.stream_sid = msg["start"]["streamSid"]
                print("STREAM START:", transport.stream_sid)
                continue

            if event == "stop":
                print("STREAM STOP")
                await pipeline.flush_buffer()
                break

            if event == "media":
                ulaw = msg["media"]["payload"]
                pcm_in = ulaw_b64_to_pcm16(ulaw)
                # feed pipeline (non-blocking)
                await pipeline.accept_incoming(pcm_in)
    except Exception as e:
        print("ERROR in websocket loop:", e)
    finally:
        # don't call ws.close(); Twilio handles closing
        print("WebSocket closed")

app.include_router(router)