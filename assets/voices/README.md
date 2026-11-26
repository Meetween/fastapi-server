Voice samples, one `<voice_id>.wav` per voice of `AGENTAR_VOICES` (af_heart, af_bella, am_fenrir, am_michael).

Rendered once with the worker Kokoro TTS (16 kHz, mono, 16 bit) from the sentence "Hello, I am an
assistant for this meeting. This is how my voice sounds." and committed here. A voice listed in
`AGENTAR_VOICES` without a file answers 404 on `GET /api/agentar/voices/<voice_id>/sample`.
