import torch
from utils.utils import preprocess_audio, postprocess_output

def transcribe(model, audio):
    audio = preprocess_audio(audio)
    model.eval()
    with torch.no_grad():
        output = model(audio)
    transcription = postprocess_output(output)
    return transcription
