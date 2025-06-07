from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import torchaudio
import openunmix
import os
import tempfile
import torch

app = FastAPI(title="Open-Unmix API")

# Load models ONCE at startup (saves memory + time)
models = {
    "vocals": openunmix.umxl(pretrained=True).eval(),
    "drums": openunmix.umxl(pretrained=True, target="drums").eval(),
    "bass": openunmix.umxl(pretrained=True, target="bass").eval(),
    "other": openunmix.umxl(pretrained=True, target="other").eval(),
}

@app.post("/separate/{target}")
async def separate_audio(
    target: str, 
    file: UploadFile = File(..., description="Upload an audio file (MP3/WAV)")
):
    if target not in models:
        raise HTTPException(status_code=400, detail="Target must be: vocals, drums, bass, or other")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    # Process audio
    try:
        audio, rate = torchaudio.load(input_path)
        estimates = openunmix.separate(audio, rate, model=models[target])
        output_path = f"separated_{target}.wav"
        torchaudio.save(output_path, estimates[target], rate)
        return FileResponse(output_path, media_type="audio/wav")
    finally:
        os.unlink(input_path)  # Cleanup
