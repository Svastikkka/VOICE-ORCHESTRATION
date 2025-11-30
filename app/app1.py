import os
import json
import base64
import audioop
import aiohttp
import asyncio
from fastapi import FastAPI, WebSocket, APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
IN_RATE = 8000
OUT_RATE = 8000

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

def pcm16_to_ulaw_b64(pcm16: bytes) -> str:
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return base64.b64encode(ulaw).decode()

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
class ElevenLabsTTS:
    async def run(self, text: str) -> bytes:
        if not EL_KEY or not EL_VOICE:
            # Return 0.5s silence at 8k
            return b'\x00' * (IN_RATE // 2 * 2)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{ELEVEN_URL}/{EL_VOICE}",
                    json={"text": text, "model_id": "eleven_multilingual_v2"},
                    headers={"xi-api-key": EL_KEY},
                    timeout=20,
                ) as resp:
                    raw = await resp.read()
            except Exception as e:
                print("ElevenLabs request failed:", e)
                return b'\x00' * (IN_RATE // 2 * 2)

        # ElevenLabs might return WAV/RIFF or raw pcm - detect and extract PCM16 frames
        pcm16 = None
        try:
            if raw[:4] == b'RIFF' or raw[:4] == b'RIFX':
                # WAV -> extract frames
                with io.BytesIO(raw) as buf:
                    with wave.open(buf, 'rb') as wf:
                        frames = wf.readframes(wf.getnframes())
                        sample_rate = wf.getframerate()
                        chans = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        if sampwidth != 2:
                            # convert to 16-bit if necessary
                            frames = audioop.lin2lin(frames, sampwidth, 2)
                        # If stereo, downmix to mono
                        if chans > 1:
                            frames = audioop.tomono(frames, 2, 0.5, 0.5)
                        pcm16 = frames
                        src_rate = sample_rate
            else:
                # Not WAV — assume raw PCM16 at 16000Hz (common)
                pcm16 = raw
                src_rate = 16000
        except Exception as e:
            print("Error parsing ElevenLabs audio:", e)
            pcm16 = raw
            src_rate = 16000

        pcm16 = ensure_even_bytes(pcm16)

        # If source is not 16k, attempt to resample from src_rate -> 16000 first (rare)
        # For safety we handle expected 16000 -> 8000 downsample
        try:
            if src_rate != 8000:
                # downsample 16k -> 8k
                pcm8 = audioop.ratecv(pcm16, 2, 1, src_rate, 8000, None)[0]
            else:
                pcm8 = pcm16
        except Exception as e:
            print("audioop.ratecv failed:", e)
            # fallback: silence
            pcm8 = b'\x00' * (IN_RATE // 2 * 2)

        pcm8 = ensure_even_bytes(pcm8)
        return pcm8


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
        self.MIN_MS = 250  # threshold to send to STT
        self.BYTES_PER_MS = (IN_RATE * 2) // 1000  # 16 bytes per ms for 8k PCM16
        self.outgoing_lock = asyncio.Lock()
        self.processing_tasks = set()

    async def accept_incoming(self, pcm16: bytes):
        """Called for each incoming media event. Non-blocking: spawns background work when buffer full."""
        self.buffer += pcm16
        threshold = self.MIN_MS * self.BYTES_PER_MS
        if len(self.buffer) < threshold:
            return  # wait for more audio

        # grab chunk and clear buffer
        chunk = self.buffer
        self.buffer = b""

        # spawn a background job to process chunk so websocket loop keeps receiving audio
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
            pcm_out = await self.tts.run(reply)  # pcm at 8k
            # split into 20ms frames for lower-latency playback
            frame_ms = 20
            bytes_per_frame = (8000 * 2) * frame_ms // 1000  # samples * 2 bytes
            # send frames sequentially
            async with self.outgoing_lock:
                for frame in split_frames(pcm_out, bytes_per_frame):
                    if not frame:
                        continue
                    payload = pcm16_to_ulaw_b64(frame)
                    await self.transport.send_media_payload(payload)
                    # short sleep so Twilio plays back as streaming frames instead of one giant chunk
                    await asyncio.sleep(frame_ms / 1000.0)
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
    PUBLIC_WSS = os.getenv("PUBLIC_WEBSOCKET_URL", "wss://bot-orchestration.ngrok.dev/twilio-stream")
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