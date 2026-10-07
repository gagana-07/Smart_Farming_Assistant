from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/mms-tts-hin",
    local_dir="models/mms-tts-hin"
)

print("Hindi TTS model downloaded to models/mms-tts-hin")