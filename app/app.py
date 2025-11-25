import os
import json
import base64
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

# ----------------------------
#  ENV VARS
# ----------------------------
PUBLIC_URL = os.getenv("PUBLIC_HTTPS_URL", "https://bot-orchestration.ngrok.dev")
PUBLIC_WS  = f"wss://{PUBLIC_URL.replace('https://','')}/ws/twilio"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY")

# ----------------------------
#  MINIMAL MOCK COMPONENTS
#  (Replace with real Pipecat classes)
# ----------------------------
class DeepgramSTT:
    async def transcribe(self, pcm_data):
        # TODO: Call Deepgram Live
        return "hello"         

class OpenAIRealtime:
    async def respond(self, text):
        # TODO: Call OpenAI Realtime
        return "Hello from AI"

class ElevenLabsTTS:
    async def synthesize(self, text):
        # TODO: Call ElevenLabs TTS → return PCM μ-law bytes
        return b"\x00" * 320    # Dummy 20ms frame

# ----------------------------
#  VOICE PIPELINE
# ----------------------------
class Pipeline:
    def __init__(self):
        self.stt = DeepgramSTT()
        self.llm = OpenAIRealtime()
        self.tts = ElevenLabsTTS()

    async def process_audio(self, pcm_data):
        text = await self.stt.transcribe(pcm_data)
        print(f"STT: {text}")

        reply = await self.llm.respond(text)
        print(f"LLM: {reply}")

        pcm_out = await self.tts.synthesize(reply)
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
    When someone calls the Twilio number, Twilio hits this.
    We tell it to start streaming audio to the websocket.
    """
    resp = VoiceResponse()
    stream = Stream(url=PUBLIC_WS)
    resp.append(Connect().append(stream))
    return PlainTextResponse(str(resp), media_type="application/xml")


# ----------------------------
#  TWILIO WEBSOCKET HANDLER
# ----------------------------
@app.websocket("/ws/twilio")
async def twilio_websocket(ws: WebSocket):
    """
    Twilio sends call audio → we decode μ-law PCM → pass to pipeline →
    pipeline returns PCM audio → send back to Twilio.
    """
    await ws.accept()

    pipeline = Pipeline()

    while True:
        message = await ws.receive_text()
        data = json.loads(message)

        # Twilio audio chunk
        if "media" in data:
            pcm_in = base64.b64decode(data["media"]["payload"])

            # pipeline -> text -> llm -> tts
            pcm_out = await pipeline.process_audio(pcm_in)

            # Send response back to Twilio
            await ws.send_text(json.dumps({
                "event": "media",
                "media": {
                    "payload": base64.b64encode(pcm_out).decode()
                }
            }))

        # End
        elif "stop" in data:
            print("Twilio stream ended")
            break


# ----------------------------
#  MAIN
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
