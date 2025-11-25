import json
import base64
import audioop
from fastapi import FastAPI, WebSocket, APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream


# ============================================
# AUDIO SETTINGS
# ============================================
IN_RATE = 8000     # Twilio μ-law input rate
OUT_RATE = 8000    # Keeping same for simplicity


# ============================================
# AUDIO HELPERS
# ============================================
def ulaw_b64_to_pcm16(b64_data: str) -> bytes:
    """Convert incoming Base64 μ-law → PCM16"""
    ulaw_bytes = base64.b64decode(b64_data)
    return audioop.ulaw2lin(ulaw_bytes, 2)


def pcm16_to_ulaw_b64(pcm16: bytes) -> str:
    """Convert PCM16 → Base64 μ-law"""
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return base64.b64encode(ulaw).decode()


# ============================================
# FAKE STT / TTS
# (Replace with Deepgram + ElevenLabs later)
# ============================================
async def fake_stt(_pcm):
    return "hello"


async def fake_tts(_text):
    # Return 0.5s silence PCM16
    return b"\x00" * (IN_RATE // 2 * 2)


# ============================================
# FASTAPI APP
# ============================================
app = FastAPI()
router = APIRouter()


# ============================================
# 1️⃣ TWILIO WEBHOOK
# Incoming voice call lands here
# ============================================
@app.post("/incoming_call")
async def incoming_call(request: Request):
    """
    Twilio hits this when a call begins.
    Respond with TwiML to open a WebSocket stream.
    """

    # IMPORTANT — must be WSS and public
    PUBLIC_WSS = "wss://bot-orchestration.ngrok.dev/twilio-stream"

    resp = VoiceResponse()
    connect = Connect()

    # FIX: Twilio requires valid track
    # connect.append(Stream(url=PUBLIC_WSS, track="inbound_audio"))
    connect.append(Stream(url=PUBLIC_WSS))

    resp.append(connect)

    print("Returning TwiML to Twilio...")

    # Return XML correctly
    return Response(str(resp), media_type="application/xml")


# ============================================
# 2️⃣ TWILIO WEBSOCKET STREAM
# Bi-directional audio handler
# ============================================
@router.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    print("Twilio WebSocket connected")

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            event = data.get("event")

            # ---------------------------
            # CALL EVENTS
            # ---------------------------
            if event == "start":
                print("Call started")
                continue

            if event == "stop":
                print("Call stopped")
                break

            # ---------------------------
            # MEDIA EVENT (audio)
            # ---------------------------
            if event == "media":
                media_payload = data["media"]["payload"]

                # μ-law → PCM
                pcm_in = ulaw_b64_to_pcm16(media_payload)

                # STT (fake)
                text = await fake_stt(pcm_in)
                print("Caller said:", text)

                # TTS (fake)
                pcm_out = await fake_tts(text)

                # PCM → μ-law
                payload_out = pcm16_to_ulaw_b64(pcm_out)

                # Send audio back
                await ws.send_text(json.dumps({
                    "event": "media",
                    "media": {"payload": payload_out}
                }))

                print("Sent audio back to caller")

    except Exception as e:
        print("WebSocket error:", e)

    finally:
        print("WebSocket closed")
        await ws.close()


app.include_router(router)
