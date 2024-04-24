import sounddevice as sd
import openai
import os
from dotenv import load_dotenv
from pydub import AudioSegment

def initialize_openai_api():
    # Load environment variables from .env file
    load_dotenv()
    # Initialize OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Record audio
def record_audio(duration, samplerate=44100):
    print("Starting recording...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()
    audio = AudioSegment(data=audio_data.tobytes(), sample_width=2, frame_rate=samplerate, channels=2)
    audio.export("audio.mp3", format="mp3")
    print("Recording complete.")

# Transcribe audio using Whisper
def transcribe_with_whisper():
    audio_file = open("audio.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript

if __name__ == "__main__":
    initialize_openai_api()
    # Record for 5 seconds
    record_audio(5)
    # Transcribe the audio
    transcript = transcribe_with_whisper()
    print("Transcript:", transcript)
