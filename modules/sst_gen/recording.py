import pyaudio
from pydub import AudioSegment
import tkinter as tk
from threading import Thread
import os 
import time
from pathlib import Path

from tools import transcribe_speech
from utils import output_dir

# Audio settings
chunk = 1024  # Record in chunks
format = pyaudio.paInt16  # 16 bits per sample
channels = 1  # Mono
sample_rate = 44100  # Sample rate
p = pyaudio.PyAudio()

frames = []
recording = False

def record():
    global frames, stream
    while recording:
        data = stream.read(chunk)
        frames.append(data)

def toggle_recording(event=None):
    global recording, stream
    if not recording:
        # Start recording
        frames.clear()
        stream = p.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
        print("Recording...")
        recording = True
        Thread(target=record).start()
    else:
        # Stop recording
        recording = False
        stream.stop_stream()
        stream.close()
        print("Recording stopped.")

        # Save and transcribe asynchronously
        Thread(target=save_and_transcribe, args=("user_input",)).start()

def save_and_transcribe(filename="audio_input"):
    full_path = str(output_dir() / (filename + ".mp3"))  # Ensure ".mp3" is added to the filename
    if frames:
        audio_data = b''.join(frames)
        audio = AudioSegment(
            data=audio_data, sample_width=p.get_sample_size(format),
            frame_rate=sample_rate, channels=channels
        )
        audio.export(full_path, format="mp3")
        # print("Recording saved as " + full_path)

        # Short delay to ensure file is written
        time.sleep(2)

        if os.path.exists(full_path):
            transcript = transcribe_speech(full_path)
            if transcript:
                print("Transcription:", transcript)
                return transcript
            else:
                print("Transcription failed.")
                
def main():
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Audio Recorder")
    root.geometry("300x200")  # Set the size of the window

    # Bind SHIFT+W to toggle_recording function
    root.bind("<Shift-W>", toggle_recording)

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
