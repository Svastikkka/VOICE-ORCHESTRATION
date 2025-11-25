# OSS

### What is Pipecat?
- A real-time pipeline framework that connects audio streams, transcription (STT), LLM responses, and text-to-speech in streaming mode.

### Everything is a Frame
- Audio pieces
- LLM tokens
- Text messages
- Metadata

### Supports MANY Providers
- OpenAI
- Deepgram
- Gemini
- ElevenLabs
- Whisper

### Flow
```
Audio Input → VAD → STT → Text → LLM → Text → TTS → Audio Output
```
Note: Each arrow is a Processor
### Structure

- /core/frames
    - Defines the basic unit of communication → Frame
- /core/processors
    - Processors do transformations on frames:
        - converting audio
        - aggregating tokens
        - filtering speech
        - Note: A pipeline is just a chain of processors.
- /core/pipeline
    - pipeline.py
    - parallel_pipeline.py
    - runner.py
    - It: 
        - Takes processors
        - Executes them in order
        - Handles async streaming
- /core/services: All the cloud integrations live here
    - STT service implementations
    - LLM service implementations
    - TTS service implementations
- /core/transports: How audio gets in & out
    - Local microphone
    - WebRTC
    - WS client/server
    - Twilio
    - LiveKit
    
