import os
import sys
import pyaudio
import wave
import threading
from pydub import AudioSegment
import numpy as np
import noisereduce as nr
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import speech_recognition as sr
import pvporcupine, pvcobra
import time
import librosa

# Load environment variables
load_dotenv()
api_key = os.getenv("PORCUPINE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Audio recording parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1             # Mono audio
RATE = 16000             # Sample rate (16 kHz) 44100 16000
CHUNK = 512             # Number of frames per buffer 1024 512
RECORD_SECONDS = 4       # Adjust based on how long you want to record
SAMPLE_WIDTH = pyaudio.PyAudio().get_sample_size(FORMAT)  # Set once and use as a constant

# Initialize PyAudio
p = pyaudio.PyAudio()

def list_audio_devices():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    p.terminate()

def speech_recognizer_google():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(">>> Google listening for  speech ..")
        audio_data = recognizer.listen(source)
        print(">>> Recognized and transcribing...")
        try:
            # Using Google Web Speech API
            return recognizer.recognize_google(audio_data)
        except (sr.UnknownValueError, sr.RequestError):
            return None

def record_noise_sample(file_path=str(project_root / 'output/tmp/noise_sample.wav')):
    print(">>> Recording background noise sample...")
    pp = pyaudio.PyAudio()  # Use a local instance instead of the global 'p'
    try:
        stream = pp.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
            output=False,
            input_device_index=1
        )
    except Exception as e:
        print("Error opening audio stream:", e)
        return None

    frames = []

    try:
        # Record audio frames
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK)
            except Exception as e:
                print(f"Error reading audio frame: {e}")
                continue
            frames.append(data)
            print("Recording background noise sample...")
    except Exception as e:
        print("Error recording audio frames:", e)
        return None
    finally:
        # Close the audio stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

    try:
        # Save the recording as WAV
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(44100)
            wf.writeframes(b''.join(frames))
    except Exception as e:
        print("Error saving WAV file:", e)
        return None

    print(f"Background noise sample saved to {file_path}")
    return file_path

def test_cobra():
    print(">>> Testing Cobra with continuous recording and silence detection...")
    
    cobra = pvcobra.create(access_key=api_key)

    # Load the recorded noise sample for noise reduction
    noise_sample_path = str(project_root / 'output/tmp/noise_sample.wav')
    noise_audio = librosa.load(noise_sample_path, sr=RATE)  # Load with its own sample rate

    # Set up the PyAudio stream
    pp = pyaudio.PyAudio()
    stream = pp.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,  # Use the same sample rate as the noise sample
                    input=True,
                    frames_per_buffer=CHUNK)

    is_voice_active = False  # To track when voice is actively being detected
    silence_frames = 0  # Count silence frames after voice has been detected
    MAX_SILENCE_FRAMES = 35  # Threshold of silence frames to stop recording

    try:
        print("Listening... Press Ctrl+C to stop.")
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)

            # Apply noise reduction using the loaded noise sample
            reduced_noise_pcm = nr.reduce_noise(y=pcm, y_noise=noise_audio, sr=RATE)

            # Determine if current frame is voice or silence based on threshold
            is_voice = cobra.process(reduced_noise_pcm) > 0.3 # Adjust threshold as needed

            # print(f"Processing... Is voice: {is_voice}")  # Debug output, prints confidence level
            
            if is_voice:
                if not is_voice_active:
                    is_voice_active = True  # Voice started
                    print("Voice detected!")
                silence_frames = 0  # Reset silence counter whenever voice is detected
            else:
                if is_voice_active:
                    silence_frames += 1
                    print(f"Silence frame count: {silence_frames}")
                    if silence_frames >= MAX_SILENCE_FRAMES:  # Stop recording after enough silence
                        print("Stopping recording after sustained silence.")
                        is_voice_active = False  # Reset to detect new voice activity
                        silence_frames = 0  # Reset silence counter for next voice detection

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        pp.terminate()
        cobra.delete()

def record_audio_cobra():
    print(">>> Recording audio with Voice Activation Detection...")
    cobra = pvcobra.create(access_key=api_key)

    # Load the recorded noise sample for noise reduction
    noise_sample_path = str(project_root / 'output/tmp/noise_sample.wav')
    noise_audio = librosa.load(noise_sample_path, sr=RATE)  # Load with its own sample rate

    # Set up the PyAudio stream
    pp = pyaudio.PyAudio()
    stream = pp.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,  # Use the same sample rate as the noise sample
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    is_voice_active = False  # To track when voice is actively being detected
    silence_frames = 0  # Count silence frames after voice has been detected
    MAX_SILENCE_FRAMES = 48  # Threshold of silence frames to stop recording

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)

            # Apply noise reduction using the loaded noise sample
            reduced_noise_pcm = nr.reduce_noise(y=pcm, y_noise=noise_audio, sr=RATE)

            # Determine if current frame is voice or silence based on threshold
            is_voice = cobra.process(reduced_noise_pcm) > 0.3 # Adjust threshold as needed

            if is_voice:
                if not is_voice_active:
                    is_voice_active = True  # Voice started
                    print(">>> Voice detected, starting Recording!")
                silence_frames = 0  # Reset silence counter whenever voice is detected
                frames.append(data)  # Append voice frames to the recording buffer
            else:
                if is_voice_active:
                    silence_frames += 1
                    # print(f"Silence frame count: {silence_frames}")
                    if silence_frames >= MAX_SILENCE_FRAMES:  # Stop recording after enough silence
                        print("Stopping recording after sustained silence.")
                        is_voice_active = False  # Reset to detect new voice activity
                        silence_frames = 0  # Reset silence counter for next voice detection
                        break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        cobra.delete()

    # Save the recording as MP3
    audio_path = project_root / 'output/tmp/audio_cobra_recording.mp3'
    save_audio_mp3(frames, audio_path)

    return audio_path

def record_audio():
    print(">>> Recording audio...")
    pp = pyaudio.PyAudio()
    stream = pp.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pp.terminate()

    # Save the recording as MP3
    audio_path = project_root / 'output/tmp/audio_recording.mp3'
    save_audio_mp3(frames, audio_path)
    
    return audio_path

def save_audio_mp3(frames, file_path):
    """ Save the audio frames to an MP3 file. """
    print(f">>> Saving audio recording on path: {file_path}")
    try:
        # Concatenate all frames into a single byte string
        audio_data = b''.join(frames)
        
        # Create an audio segment
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=SAMPLE_WIDTH,
            frame_rate=RATE,
            channels=CHANNELS
        )
        
        # Export the audio segment to an MP3 file
        audio_segment.export(file_path, format="mp3")
    except Exception as e:
        print(f"Failed to save MP3 file: {e}")

def transcribe_whisper(audio_path):
    # print(f"Transcribing file: {audio_path}")
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
    except Exception as e:
        print(f"Failed to transcribe audio: {e}")
        return None

def listening_loop():
    keyword_path = project_root / "src/local_models/picovoice/heyarx_mac.ppn"
    print("Checking keyword path:", keyword_path)
    
    if not keyword_path.exists():
        print("Keyword file does not exist at:", keyword_path)
        return

    # porcupine = pvporcupine.create(access_key=api_key, keyword_paths=[keyword_path], sensitivities=[1.0])
    porcupine = pvporcupine.create(access_key=api_key, keywords=["computer"], sensitivities=[1.0])
    pp = pyaudio.PyAudio()
    audio_stream = pp.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)
        
    print(">>> Listening for wake word...")
    
    try:
        while True:
            data = audio_stream.read(porcupine.frame_length)
            pcm = np.frombuffer(data, dtype=np.int16)
            if porcupine.process(pcm) >= 0:
                # try:
                #     print(">>> Wake word detected, recording audio...")
                #     transcript = speech_recognizer_google()
                #     print(">>> Transcription: " + transcript)
                #     return transcript
                # except Exception as e:
                print(">>> Wake word detected, kicking recording audio...")
                audio_path = record_audio_cobra()
                time.sleep(3)  # Sleep for 3 seconds to save the audio
                print(">>> Transcribing audio...")
                transcript = transcribe_whisper(audio_path)
                if transcript:
                    print(">>> Fallback transcription: " + transcript)
                    return transcript
                else:
                    print("Failed to transcribe audio.")
    finally:
        audio_stream.close()
        pp.terminate()
        porcupine.delete()
        
def main():
    list_audio_devices()
    
    record_noise_sample()
    
    # Start the listening thread
    listening_thread = threading.Thread(target=listening_loop)
    listening_thread.start()
    # ---------------------
    listening_thread.join()
    
    
if __name__ == "__main__":
    main()
