import os
import json
import base64
import audioop
import aiohttp
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

    async def send_audio(self, payload_b64: str):
        await self.ws.send_text(json.dumps({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": payload_b64}
        }))


# ------------------------
#  REAL STT SERVICE (Deepgram)
# ------------------------
# ------------------------
#  REAL STT SERVICE (Deepgram)
# ------------------------
class DeepgramSTT:
    async def run(self, pcm16: bytes) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                DEEPGRAM_URL,
                data=pcm16,
                headers={
                    "Content-Type": "audio/pcm",
                    "Authorization": f"Token {DG_KEY}",
                },
                params={"model": "nova-2", "encoding": "linear16", "sample_rate": 8000},
            ) as resp:
                # 1. Check for non-200 HTTP status codes
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"Deepgram HTTP Error {resp.status}: {error_text}")
                    return ""
                    
                out = await resp.json()
                
                # 2. Check for the missing 'results' key
                if "results" not in out:
                    print(f"Deepgram Response Missing 'results' Key: {out}")
                    return ""
                
                try:
                    text = out["results"]["channels"][0]["alternatives"][0]["transcript"]
                    return text if text else ""
                except (KeyError, IndexError) as e:
                    # 3. Handle structure errors (e.g., missing channels/alternatives)
                    print(f"Deepgram Structure Error ({e}): {out}")
                    return ""

# ------------------------
# REAL LLM (OpenAI GPT-4o-mini)
# ------------------------
class OpenAILLM:
    async def run(self, text: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Speak concisely. No emojis."},
                {"role": "user", "content": text},
            ]
        )
        return response.choices[0].message.content or ""

# ------------------------
# REAL TTS (ElevenLabs)
# ------------------------
class ElevenLabsTTS:
    async def run(self, text: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ELEVEN_URL}/{EL_VOICE}",
                json={"text": text, "model_id": "eleven_multilingual_v2"},
                headers={"xi-api-key": EL_KEY},
            ) as resp:
                audio = await resp.read()

        # ElevenLabs returns 16kHz PCM16 → Downsample to 8k for Twilio
        pcm8 = audioop.ratecv(audio, 2, 1, 16000, 8000, None)[0]
        return pcm8


# ------------------------
# Pipeline (Pipecat Style)
# ------------------------
class VoicePipeline:
    def __init__(self, stt, llm, tts):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.buffer = b""
        self.MIN_MS = 250   # process only after 250ms of audio
        self.BYTES_PER_MS = (8000 * 2) // 1000  # 16 bytes per ms for 8kHz PCM16

    async def process_audio(self, pcm16: bytes) -> bytes:
        self.buffer += pcm16

        # If not enough audio, return silence
        if len(self.buffer) < self.MIN_MS * self.BYTES_PER_MS:
            return b"\x00" * (IN_RATE // 2 * 2)

        # Take buffered audio and clear
        chunk = self.buffer
        self.buffer = b""

        text = await self.stt.run(chunk)
        print("STT:", text)

        if not text.strip():
            return b"\x00" * (IN_RATE // 2 * 2)

        reply = await self.llm.run(text)
        print("LLM:", reply)

        pcm_out = await self.tts.run(reply)
        return pcm_out


# ======================================================
# FastAPI Setup
# ======================================================
app = FastAPI()
router = APIRouter()

@app.post("/incoming_call")
async def incoming_call(request: Request):
    PUBLIC_WSS = "wss://bot-orchestration.ngrok.dev/twilio-stream"
    resp = VoiceResponse()
    connect = Connect()
    connect.append(Stream(url=PUBLIC_WSS))
    resp.append(connect)
    return Response(str(resp), media_type="application/xml")


# ======================================================
# WebSocket Handler
# ======================================================
@router.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    transport = TwilioTransport(ws)
    pipeline = VoicePipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=ElevenLabsTTS()
    )

    print("Twilio WebSocket connected")

    try:
        while True:
            data = await transport.receive()
            event = data.get("event")

            if event == "start":
                transport.stream_sid = data["start"]["streamSid"]
                print("STREAM START:", transport.stream_sid)
                continue

            if event == "stop":
                print("STREAM STOP")
                break

            if event == "media":
                pcm_in = ulaw_b64_to_pcm16(data["media"]["payload"])
                pcm_out = await pipeline.process_audio(pcm_in)
                payload_out = pcm16_to_ulaw_b64(pcm_out)
                await transport.send_audio(payload_out)

    except Exception as e:
        print("ERROR:", e)

    finally:
        # Do NOT call ws.close()
        print("WebSocket closed")


app.include_router(router)