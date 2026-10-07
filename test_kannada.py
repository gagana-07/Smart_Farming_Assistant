from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="vasista22/whisper-kannada-medium",
    chunk_length_s=30,
    device=-1,   # CPU
)
asr.model.config.forced_decoder_ids = asr.tokenizer.get_decoder_prompt_ids(
    language="kn", task="transcribe"
)

print(asr("audio/input.wav")["text"])