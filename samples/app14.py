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

#   Google STT imports
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

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
LLM_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
PROJECT_ID = os.getenv("PROJECT_ID") # For GCP only

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
class GoogleStreamingSTT:
    def __init__(self, on_utterance_complete):
        self.on_utterance_complete = on_utterance_complete
        self.client = SpeechAsyncClient(
            client_options=ClientOptions(api_endpoint="speech.googleapis.com")
        )
        self.audio_queue = asyncio.Queue()
        self.connected = False
        self._main_task = None
        self.buffer = bytearray()
        self.buffer_threshold = 1600
        self.stream_start_time = None
        self.first_interim_time = None
        self.first_final_time = None
        # Using "_" for the default/inline recognizer is most reliable
        self.recognizer = f"projects/{PROJECT_ID}/locations/global/recognizers/telephony-recognizer-global"

    async def connect(self):
        if self.connected: return
        self.connected = True
        self._main_task = asyncio.create_task(self._infinite_loop())
        print("[Google STT] Connection task started.")

    async def _infinite_loop(self):
        while self.connected:
            try:
                await self._run_single_stream_v2()
            except Exception as e:
                print(f"[Google STT] Stream error: {e}. Restarting in 1s...")
                await asyncio.sleep(1)

    async def _run_single_stream_v2(self):
        self.stream_start_time = time.time()
        self.first_interim_time = None
        self.first_final_time = None
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
                audio_channel_count=1,
            ),
            language_codes=["en-US"],
            model="telephony", 
            features=cloud_speech.RecognitionFeatures(
                profanity_filter=True,
                enable_automatic_punctuation=True
            ),
        )

        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(interim_results=True)
        )

        async def request_generator():
            # 1. First yield MUST be the config
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=self.recognizer,
                streaming_config=streaming_config
            )
            
            while self.connected:
                chunk = await self.audio_queue.get()
                if chunk is None: break
                # 2. FIX: Field name is 'audio', not 'audio_content'
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

        # 3. Process responses
        responses = await self.client.streaming_recognize(requests=request_generator())

        async for response in responses:
            if not self.connected: break
            if not response.results: continue

            result = response.results[0]
            if not result.alternatives: continue

            transcript = result.alternatives[0].transcript
            if not result.is_final:
                if self.first_interim_time is None:
                    self.first_interim_time = time.time()
                    print(f"[STT] TTFT (audio → first interim): "
                        f"{self.first_interim_time - self.stream_start_time:.3f}s")
                print(f"STT (Interim): {transcript}")
            else:
                if self.first_final_time is None:
                    self.first_final_time = time.time()
                    print(f"[STT] TTFB (audio → first final): "
                        f"{self.first_final_time - self.stream_start_time:.3f}s")
                print(f"STT (Final): {transcript}")
                await self.on_utterance_complete(transcript)

    async def send_audio(self, mulaw: bytes):
        if self.connected:
            self.buffer.extend(mulaw)
            if len(self.buffer) >= self.buffer_threshold:
                await self.audio_queue.put(bytes(self.buffer))
                self.buffer.clear()

    async def close(self):
        self.connected = False
        await self.audio_queue.put(None)
        if self._main_task:
            self._main_task.cancel()
        print("[Google STT] Connection closed.")
# ------------------------
# Custom LLM (Streaming)
# ------------------------
class CustomLLM:
    def __init__(self, prompt_file_name="prompt.txt"):
        self.client = AsyncOpenAI(api_key="any", base_url=f"{LLM_URL}")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(current_dir, prompt_file_name)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "troubleshoot_printer_issue",
                    "description": "Diagnose and provide troubleshooting steps for printer issues",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "printer_model": {
                                "type": "string",
                                "description": "Model of the printer if known"
                            },
                            "issue_type": {
                                "type": "string",
                                "description": "Type of printer issue (e.g., not printing, paper jam, offline, poor quality)"
                            },
                            "error_message": {
                                "type": "string",
                                "description": "Any error message displayed on the printer or computer"
                            }
                        },
                        "required": ["issue_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "troubleshoot_wifi_issue",
                    "description": "Diagnose and provide troubleshooting steps for WiFi connectivity issues",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device_type": {
                                "type": "string",
                                "description": "Type of device (laptop, desktop, mobile)"
                            },
                            "operating_system": {
                                "type": "string",
                                "description": "Operating system of the device"
                            },
                            "issue_description": {
                                "type": "string",
                                "description": "Description of the WiFi issue (e.g., cannot connect, slow speed, frequent disconnect)"
                            }
                        },
                        "required": ["issue_description"]
                    }
                }
            }
        ]
        with open(prompt_file_path, "r") as f:
            self.system_prompt = f.read()
        self.system_prompt_added = False

    async def execute_tool(self, name, arguments):
        print(f"\n🛠️ TOOL EXECUTION STARTED: {name}")
        print(f"🛠️ Arguments: {arguments}")

        # ==========================
        # PRINTER TROUBLESHOOTING
        # ==========================
        if name == "troubleshoot_printer_issue":
            issue = arguments.get("issue_type", "").lower()

            steps = []

            if "offline" in issue:
                steps = [
                    "Check if the printer is powered on.",
                    "Ensure the printer is connected to the network.",
                    "Restart the printer.",
                    "Restart your computer.",
                    "Set printer as default in settings."
                ]

            elif "paper jam" in issue:
                steps = [
                    "Turn off the printer.",
                    "Open the paper tray.",
                    "Carefully remove jammed paper.",
                    "Check for small torn pieces inside.",
                    "Restart the printer."
                ]

            else:
                steps = [
                    "Restart the printer.",
                    "Check cable or WiFi connection.",
                    "Reinstall printer drivers."
                ]

            return {
                "status": "success",
                "category": "printer",
                "recommended_steps": steps[:2]
            }

        # ==========================
        # WIFI TROUBLESHOOTING
        # ==========================
        if name == "troubleshoot_wifi_issue":
            issue = arguments.get("issue_description", "").lower()

            steps = []

            if "cannot connect" in issue:
                steps = [
                    "Restart your router.",
                    "Restart your device.",
                    "Forget and reconnect to the WiFi network.",
                    "Check if airplane mode is off.",
                    "Ensure correct password is entered."
                ]

            elif "slow" in issue:
                steps = [
                    "Move closer to the router.",
                    "Restart the router.",
                    "Disconnect unused devices.",
                    "Run a speed test."
                ]

            else:
                steps = [
                    "Restart router and device.",
                    "Check network cables.",
                    "Contact IT if issue persists."
                ]

            return {
                "status": "success",
                "category": "wifi",
                "recommended_steps": steps
            }

        return {"status": "error", "message": "Unknown tool"}

    async def run_streaming(self, conversation_history: list, on_token=None) -> str:
        messages = []
        if not self.system_prompt_added:
            messages.append({"role": "system", "content": self.system_prompt})
            self.system_prompt_added = True

        messages += conversation_history[-10:]

        start_time = time.time()
        response = await self.client.chat.completions.create(
            model=f"{LLM_MODEL}",
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            stream=True
        )

        full_reply = []
        first_token_time = None

        tool_name = None
        tool_arguments = ""

        async for chunk in response:
            delta = chunk.choices[0].delta

            # ==========================
            # 🛠️ TOOL CALL DETECTION
            # ==========================
            if delta.tool_calls:
                print("\n🚨 TOOL CALL DETECTED")

                tc = delta.tool_calls[0]

                if tc.function.name:
                    tool_name = tc.function.name
                    print(f"🛠️ Tool Name: {tool_name}")

                if tc.function.arguments:
                    tool_arguments += tc.function.arguments

                continue

            # ==========================
            # NORMAL TEXT STREAMING
            # ==========================
            token = delta.content
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttfb = first_token_time - start_time
                    print(f"[LLM] Streaming first token TTFB: {ttfb:.3f}s")
                full_reply.append(token)
                if on_token:
                    await on_token(token)

        # ==========================
        # If tool was called
        # ==========================
        if tool_name:
            print(f"\n🧠 TOOL CALL COMPLETE")
            print(f"🧠 Raw Arguments: {tool_arguments}")

            try:
                parsed_args = json.loads(tool_arguments)
            except:
                print("❌ Failed to parse tool arguments")
                parsed_args = {}

            tool_result = await self.execute_tool(tool_name, parsed_args)

            print(f"🛠️ Tool Result: {tool_result}\n")

            # Add tool call to conversation
            conversation_history.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": "tool_call_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_arguments
                    }
                }]
            })

            conversation_history.append({
                "role": "tool",
                "tool_call_id": "tool_call_1",
                "content": json.dumps(tool_result)
            })

            # 🔁 SECOND CALL to generate final natural response
            print("🔁 Calling LLM again to generate final response...")

            final_response = await self.client.chat.completions.create(
                model=f"{LLM_MODEL}",
                messages=conversation_history,
                stream=True
            )

            final_text = []
            async for chunk in final_response:
                token = chunk.choices[0].delta.content
                if token:
                    final_text.append(token)
                    if on_token:
                        await on_token(token)

            return "".join(final_text).strip()


        return "".join(full_reply).strip()



# ------------------------
# ElevenLabs TTS
# ------------------------
class ElevenLabsTTS:
    def __init__(self):
        # Change output_format to ulaw_8000
        self.uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{EL_VOICE}/stream-input"
            f"?model_id=eleven_turbo_v2_5&output_format=ulaw_8000"
        )

    async def run_streaming(self, text_iterator, on_audio_chunk):
        async with websockets.connect(self.uri) as ws:
            tts_start_time = None
            first_audio_received_time = None
            await ws.send(json.dumps({
                "text": " ", 
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": EL_KEY,
            }))

            async def listen_for_audio():
                nonlocal first_audio_received_time
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)
                        if data.get("audio"):
                            if first_audio_received_time is None and tts_start_time:
                                first_audio_received_time = time.time()
                                print(f"[TTS] TTFT (text → first audio chunk): "
                                    f"{first_audio_received_time - tts_start_time:.3f}s")
                            # This is now raw ulaw audio, ready for Twilio!
                            audio_data = base64.b64decode(data["audio"])
                            await on_audio_chunk(audio_data)
                        if data.get("isFinal"):
                            break
                    except:
                        break

            listen_task = asyncio.create_task(listen_for_audio())

            async for text in text_iterator:
                if text:
                    if tts_start_time is None:
                        tts_start_time = time.time()
                    await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))

            await ws.send(json.dumps({"text": ""}))
            await listen_task

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
        self.stt = GoogleStreamingSTT(self._on_stt_utterance)
        self.history = []
        self.first_interaction_done = False
        self.turn_start_time = None
        self.first_audio_played = False
        asyncio.create_task(self._playback_worker())

    async def start_greeting_with_delay(self):
        await asyncio.sleep(1.0)  # wait 1 second before speaking
        await self.start_greeting()

    async def start_greeting(self):
        """Triggers the bot to speak first."""
        if not self.first_interaction_done:
            self.first_interaction_done = True
            greeting = "Hello, this is Alex from IT Support. How can I help you today?"
            self.history.append({"role": "assistant", "content": greeting})
            
            # Pipe the greeting directly to TTS
            text_stream_queue = asyncio.Queue()
            async def text_gen():
                yield greeting
                yield None
            
            await self.tts.run_streaming(text_gen(), self.playback_queue.put)

    # Inside VoicePipeline
    async def _on_stt_utterance(self, text):
        # 🔥 Mark turn start (user finished speaking)
        self.turn_start_time = time.time()
        self.first_audio_played = False
        # If the bot was greeting and user interrupted, clear it
        if self.is_playing_tts:
             await self.interrupt()
             
        self.history.append({"role": "user", "content": text})
        asyncio.create_task(self._generate_and_play_response())

    async def interrupt(self):
        """Helper to stop playback immediately."""
        if self.current_tts_task:
            self.current_tts_task.cancel()
        await self.transport.clear_buffer()
        while not self.playback_queue.empty():
            self.playback_queue.get_nowait()
            self.playback_queue.task_done()

    async def _generate_and_play_response(self):
        text_stream_queue = asyncio.Queue()

        async def text_gen():
            while True:
                val = await text_stream_queue.get()
                if val is None: break
                yield val

        # Start TTS stream
        tts_task = asyncio.create_task(
            self.tts.run_streaming(text_gen(), self.playback_queue.put)
        )
        self.current_tts_task = tts_task

        token_buffer = ""
        async def on_token(token):
            nonlocal token_buffer
            token_buffer += token
            # Improved chunking: send at spaces to avoid splitting words
            if token.endswith((" ", ".", "!", "?", "\n")):
                await text_stream_queue.put(token_buffer)
                token_buffer = ""

        reply = await self.llm.run_streaming(self.history, on_token=on_token)
        
        if token_buffer:
            await text_stream_queue.put(token_buffer)
        
        await text_stream_queue.put(None) 
        self.history.append({"role": "assistant", "content": reply})
        await tts_task

    async def _playback_worker(self):
        while True:
            try:
                audio = await self.playback_queue.get()
                await self._play_audio(audio)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print("Playback worker error:", e)
            finally:
                self.is_playing_tts = False
                self.current_tts_task = None
                self.playback_queue.task_done()

    async def _play_audio(self, audio):
        self.is_playing_tts = True
        for frame in split_frames(audio, 160):

            if not self.first_audio_played:
                self.first_audio_played = True
                now = time.time()

                if self.turn_start_time:
                    latency = now - self.turn_start_time

                    print(f"[TTS] TTFB (text → first playback frame): {latency:.3f}s")
                    print(f"[E2E] User speech_final → first audio playback: {latency:.3f}s")

            await self.transport.send_media_payload(
                base64.b64encode(frame).decode()
            )
            await asyncio.sleep(0.019)

    async def accept_audio(self, mulaw_bytes):
        # Google STT V2 expects raw mulaw for ExplicitDecodingConfig.MULAW
        await self.stt.send_audio(mulaw_bytes)
        pcm = audioop.ulaw2lin(mulaw_bytes, 2)
        if self.is_playing_tts and audioop.rms(pcm, 2) > 2500:
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
                await pipeline.stt.connect()                  # start STT here only
                asyncio.create_task(pipeline.start_greeting_with_delay())  


                # Start STT immediately
                await pipeline.stt.connect()

                # Delay greeting slightly to avoid stepping on user audio
                asyncio.create_task(pipeline.start_greeting_with_delay())
            elif msg["event"] == "media":
                mulaw = base64.b64decode(msg["media"]["payload"])
                await pipeline.accept_audio(mulaw)
            elif msg["event"] == "stop":
                await pipeline.stt.close()
                break
    except Exception as e:
        print(f"WS Error: {e}")
    finally:
        print("WebSocket closed")

app.include_router(router)