# VOSS

### Everything is a Frame
- Audio pieces
- LLM tokens
- Text messages
- Metadata

### Supports MANY Providers
- OpenAI
- Deepgram
- ElevenLabs

### Flow
```
User Voice
   ↓
Twilio MediaStream (µ-law 8kHz audio)
   ↓
Decode µ-law → PCM16
   ↓
VAD (RMS-based silence detection)
   ↓
STT (Deepgram / OpenAI Whisper)
   ↓
Text recognized
   ↓
LLM (GPT)
   ↓
LLM Text Response
   ↓
TTS (ElevenLabs: wav/mp3 bytes)
   ↓
Decode ElevenLabs output to PCM16
   ↓
Convert PCM16 → µ-law 8kHz
   ↓
Split into 20ms frames (160 bytes)
   ↓
Base64 encode each frame
   ↓
Send to Twilio MediaStream
   ↓
User hears response
```
Note: Each arrow is a Processor

### Revision

- app1.py: Basic Integration with Fake TTS/STT/LLM
- app2.py: Integration with STT and Open AI LLM
- app3.py: Integration with TTS
- app4.py: Improve VAD to take complete input even if the user take pause
- app5.py: Add BARGE-IN DETECTION
- app6.py: Add MIN_UTTERANCE_MS & MAX_PAUSE_FRAMES
- app7.py: Enable STT Streaming


# How To Run

```bash
uvicorn samples.app4:app --reload
```

```bash
ngrok http --url=https://test.ngrok.app  8000
```