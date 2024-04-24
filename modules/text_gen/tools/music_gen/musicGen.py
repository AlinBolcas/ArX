from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
import threading
from pathlib import Path
import os

processor = None
model = None
models_loaded = threading.Event()

def init_models():
    model_thread = threading.Thread(target=load_models)
    model_thread.start()
    return models_loaded.wait() 

def load_models():
    global processor, model
    if processor is None:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    if model is None:
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cpu")
    models_loaded.set()

def musicGen(prompt, max_new_tokens=300):
    models_loaded.wait()  # Wait for the models to be loaded
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt", )

    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    sampling_rate = model.config.audio_encoder.sampling_rate
    
    src_path = path = Path(__file__).parent.parent / 'tmp'
    scipy.io.wavfile.write(src_path / "musicGen.mp3", rate=sampling_rate, data=audio_values[0, 0].numpy())

    return str(src_path / "musicGen.mp3")

if __name__ == "__main__":
    init_models()
    music = musicGen("I love you", 500)
    os.system(f"open {music}")
