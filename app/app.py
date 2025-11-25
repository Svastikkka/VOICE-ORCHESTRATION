import os
import json
import base64
import asyncio
import math
import uvicorn
import audioop
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

# ----------------------------
#  ENV VARS
# ----------------------------
PUBLIC_URL = os.getenv("PUBLIC_HTTPS_URL", "https://bot-orchestration.ngrok.dev")
# Twilio expects the Stream URL to be wss (websocket). Twilio will convert http -> ws.
# We'll provide a wss URL for Twilio to connect into.
PUBLIC_WS  = f"wss://{PUBLIC_URL.replace('https://','')}/ws/twilio"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY")

# ----------------------------
#  MINIMAL MOCK COMPONENTS
#  (Replace with real Pipecat / API calls later)
# ----------------------------
class DeepgramSTT:
    async def transcribe(self, pcm16_bytes, sample_rate=8000):
        """
        pcm16_bytes: raw PCM16 little-endian bytes (mono)
        sample_rate: sample rate in Hz (Twilio will send 8000)
        For test, we return a fixed text.
        """
        # In a real implementation you'd stream pcm16_bytes to Deepgram and get partial transcripts.
        return "hello"

class OpenAIRealtime:
    async def respond(self, text):
        """
        Send text to OpenAI Realtime and stream back tokens.
        For testing return a fixed reply.
        """
        return "Hello from AI"

class ElevenLabsTTS:
    async def synthesize(self, text, sample_rate=8000):
        """
        Mock TTS: generate a short waveform (PCM16) so Twilio can play it.
        Returns PCM16 bytes, mono, sample_rate Hz.

        Replace this with real ElevenLabs TTS which may return WAV/MP3/OPUS.
        If real ElevenLabs returns >8kHz, you'll need to resample to 8000 Hz before sending to Twilio.
        """
        # Simple generated tone (sine wave) for ~0.8s for audible reply
        duration_s = 0.8
        freq = 600.0  # tone frequency
        amplitude = 16000  # max 32767
        frames = int(sample_rate * duration_s)
        pcm = bytearray()
        for n in range(frames):
            sample = int(amplitude * math.sin(2 * math.pi * freq * (n / sample_rate)))
            # pack as little-endian signed 16-bit
            pcm += int(sample & 0xffff).to_bytes(2, byteorder='little', signed=False)
        return bytes(pcm)

# ----------------------------
#  VOICE PIPELINE (wiring STT -> LLM -> TTS)
# ----------------------------
class Pipeline:
    def __init__(self):
        self.stt = DeepgramSTT()
        self.llm = OpenAIRealtime()
        self.tts = ElevenLabsTTS()

    async def process_audio(self, pcm16_bytes, sample_rate=8000):
        """Take raw PCM16 bytes (mono @ sample_rate), return PCM16 reply (mono @ 8000)."""
        # 1) STT
        text = await self.stt.transcribe(pcm16_bytes, sample_rate=sample_rate)
        print(f"STT: {text}")

        # 2) LLM
        reply = await self.llm.respond(text)
        print(f"LLM: {reply}")

        # 3) TTS -> we expect PCM16 returned
        pcm_out = await self.tts.synthesize(reply, sample_rate=8000)

        # Ensure output is PCM16 @ 8000 Hz mono (our mock already is).
        return pcm_out

# ----------------------------
#  FASTAPI SETUP
# ----------------------------
app = FastAPI()

# ----------------------------
#  TWILIO INBOUND CALL WEBHOOK
# ----------------------------
@app.post("/incoming_call")
async def incoming_call(request: Request):
    """
    Twilio posts here when a call arrives. We return TwiML instructing Twilio
    to open a Media Stream to our websocket at /ws/twilio. Set track="both"
    so Twilio will play our outgoing audio back to the caller.
    """
    resp = VoiceResponse()
    # instruct Twilio to stream both inbound and outbound audio
    stream = Stream(url=PUBLIC_WS, track="both")
    connect = Connect()
    connect.append(stream)
    resp.append(connect)
    return PlainTextResponse(str(resp), media_type="application/xml")

# ----------------------------
#  Helper: µ-law <-> PCM using audioop (stdlib)
# ----------------------------
def twilio_media_payload_to_pcm16( payload_b64: str ) -> bytes:
    """
    Convert Twilio 'media.payload' (base64 of µ-law frames) -> PCM16 little-endian bytes.
    Twilio sends 8kHz µ-law by default. We convert to 16-bit PCM so STT can use it.
    """
    ulaw = base64.b64decode(payload_b64)
    # convert µ-law to 16-bit linear (signed little-endian)
    pcm16 = audioop.ulaw2lin(ulaw, 2)  # width=2 (16-bit)
    return pcm16

def pcm16_to_twilio_payload( pcm16_bytes: bytes ) -> str:
    """
    Convert PCM16 little-endian -> µ-law base64 payload for Twilio.
    """
    ulaw = audioop.lin2ulaw(pcm16_bytes, 2)  # width=2
    return base64.b64encode(ulaw).decode('ascii')

# ----------------------------
#  TWILIO WEBSOCKET HANDLER
# ----------------------------
@app.websocket("/ws/twilio")
async def twilio_websocket(ws: WebSocket):
    """
    Twilio streams JSON events over the websocket. Typical events:
      - {"event":"start", ...}
      - {"event":"media","media":{"payload":"<base64-mulaw>"}}
      - {"event":"stop", ...}

    This handler decodes incoming µ-law frames, passes them to the pipeline,
    receives PCM16 reply, encodes to µ-law and sends back to Twilio.
    """
    await ws.accept()
    pipeline = Pipeline()

    print("Twilio websocket connected")

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            event = data.get("event")
            if event == "start":
                print("Twilio stream START:", data.get("start", {}))
                continue

            if event == "media":
                try:
                    payload_b64 = data["media"]["payload"]
                    # decode µ-law -> PCM16 (signed 16-bit little-endian)
                    pcm16_in = twilio_media_payload_to_pcm16(payload_b64)

                    # Pass PCM16 to pipeline (STT -> LLM -> TTS)
                    pcm16_out = await pipeline.process_audio(pcm16_in, sample_rate=8000)

                    # pcm16_out must be PCM16 little-endian @ 8000 Hz mono for Twilio
                    # Encode to µ-law base64 and send back
                    payload_out_b64 = pcm16_to_twilio_payload(pcm16_out)

                    out_msg = json.dumps({
                        "event": "media",
                        "media": {"payload": payload_out_b64}
                    })

                    # send it back
                    await ws.send_text(out_msg)
                    print("Sent audio -> Twilio (bytes out):", len(pcm16_out))

                except Exception as e:
                    print("Error handling media:", e)
                    # continue listening for subsequent frames
                    continue

            if event == "stop":
                print("Twilio stream STOP")
                break

    except Exception as e:
        print("Twilio websocket exception:", e)
    finally:
        await ws.close()
        print("Twilio websocket closed")

# ----------------------------
#  MAIN
# ----------------------------
if __name__ == "__main__":
    print("Starting FastAPI server (app) ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
