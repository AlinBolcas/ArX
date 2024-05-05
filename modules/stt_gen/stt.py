import os
import sys
import concurrent.futures
import pyaudio
import numpy as np
import librosa
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pathlib import Path
import pvcobra
import pvporcupine
import speech_recognition as sr
import noisereduce as nr

# allow user to choose which audio device to use.. but later on, use default for now

class STT:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.porcupine_api_key = os.getenv("PORCUPINE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Project paths
        self.project_root = Path(__file__).resolve().parents[2]
        sys.path.append(str(self.project_root))
        
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 512
        self.record_seconds = 5
        self.sample_width = pyaudio.PyAudio().get_sample_size(self.format)
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

    def list_audio_devices(self):
        info = self.p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for i in range(num_devices):
            if self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                print(f"Input Device id {i} - {self.p.get_device_info_by_host_api_device_index(0, i).get('name')}")
        self.p.terminate() 

    def speech_recognizer_google(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print(">>> Google listening for speech ..")
            audio_data = recognizer.listen(source)
            print(">>> Recognized and transcribing...")
            try:
                # Using Google Web Speech API
                return recognizer.recognize_google(audio_data)
            except (sr.UnknownValueError, sr.RequestError):
                return None

    def record_noise_sample(self, file_path=None):
        if file_path is None:
            file_path = str(self.project_root / 'output/tmp/noise_sample.wav')

        print(">>> Recording background noise sample...")
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except Exception as e:
            print("Error opening audio stream:", e)
            return None

        frames = []

        try:
            # Record audio frames
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                try:
                    data = stream.read(self.chunk)
                except Exception as e:
                    print(f"Error reading audio frame: {e}")
                    continue
                frames.append(data)
                print("Recording background noise sample...")
        except Exception as e:
            print("Error recording audio frames:", e)
            return None
        finally:
            # Ensure stream is properly closed
            stream.stop_stream()
            stream.close()
            # Optionally reinitialize PyAudio instance
            self.p.terminate()
            self.p = pyaudio.PyAudio()

        # Save the recording as WAV
        try:
            self.save_audio_mp3(frames, file_path)
        except Exception as e:
            print("Error saving WAV file:", e)
            return None

        print(f"Background noise sample saved to {file_path}")
        return file_path

    def record_audio_cobra(self):
        print(">>> Recording audio with Voice Activation Detection...")
        cobra = pvcobra.create(access_key=self.porcupine_api_key)

        # Load the recorded noise sample for noise reduction
        noise_sample_path = str(self.project_root / 'output/tmp/noise_sample.wav')
        noise_audio, _ = librosa.load(noise_sample_path, sr=self.rate)  # Load with its own sample rate

        # Reinitialize PyAudio (close and open)
        if hasattr(self, 'p'):
            self.p.terminate()  # Terminate the existing PyAudio instance
        self.p = pyaudio.PyAudio()  # Reinitialize PyAudio

        # Set up the PyAudio stream
        stream = self.p.open(
            format=self.format,
            channels=1,
            rate=self.rate,  # Use the same sample rate as the noise sample
            input=True,
            frames_per_buffer=self.chunk
        )
        frames = []
        is_voice_active = False  # To track when voice is actively being detected
        silence_frames = 0  # Count silence frames after voice has been detected
        MAX_SILENCE_FRAMES = 48  # Threshold of silence frames to stop recording

        try:
            while True:
                data = stream.read(self.chunk, exception_on_overflow=False)
                pcm = np.frombuffer(data, dtype=np.int16)

                # Apply noise reduction using the loaded noise sample
                reduced_noise_pcm = nr.reduce_noise(y=pcm, y_noise=noise_audio, sr=self.rate)

                # Determine if the current frame is voice or silence based on threshold
                is_voice = cobra.process(reduced_noise_pcm) > 0.3  # Adjust threshold as needed

                if is_voice:
                    if not is_voice_active:
                        is_voice_active = True  # Voice started
                        print(">>> Voice detected, starting Recording!")
                    silence_frames = 0  # Reset silence counter whenever voice is detected
                    frames.append(data)  # Append voice frames to the recording buffer
                else:
                    if is_voice_active:
                        silence_frames += 1
                        if silence_frames >= MAX_SILENCE_FRAMES:  # Stop recording after enough silence
                            print("Stopping recording after sustained silence.")
                            is_voice_active = False  # Reset to detect new voice activity
                            silence_frames = 0  # Reset silence counter for next voice detection
                            break

        finally:
            # Properly stop and close the stream
            stream.stop_stream()
            stream.close()
            # Reinitialize PyAudio instance to avoid resource locking
            self.p.terminate()
            self.p = pyaudio.PyAudio()

        # Save the recording as MP3
        audio_path = self.project_root / 'output/tmp/audio_cobra_recording.mp3'
        self.save_audio_mp3(frames, audio_path)

        return audio_path

    def record_audio(self):
        print(">>> Recording audio...")
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        frames = []
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        # Close the stream when done
        stream.stop_stream()
        stream.close()
        # Terminate PyAudio to free up the device
        self.p.terminate()
        self.p = pyaudio.PyAudio()

        # Save the recording as MP3
        audio_path = self.project_root / 'output/tmp/audio_recording.mp3'
        self.save_audio_mp3(frames, audio_path)

        return audio_path

    def save_audio_mp3(self, frames, file_path):
        """ Save the audio frames to an MP3 file. """
        print(f">>> Saving audio recording on path: {file_path}")
        try:
            # Concatenate all frames into a single byte string
            audio_data = b''.join(frames)
            
            # Create an audio segment
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=self.sample_width,
                frame_rate=self.rate,
                channels=self.channels
            )
            
            # Export the audio segment to an MP3 file
            audio_segment.export(file_path, format="mp3")
        except Exception as e:
            print(f"Failed to save MP3 file: {e}")

    def transcribe_whisper(self, audio_path):
        # print(f"Transcribing file: {audio_path}")
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcript.text
        except Exception as e:
            print(f"Failed to transcribe audio: {e}")
            return None

    def listen_for_wake_word(self):
        keyword_path = self.project_root / "src/local_models/picovoice/heyarx_mac.ppn"
        # print("Checking keyword path:", keyword_path)
        
        if not keyword_path.exists():
            print("Keyword file does not exist at:", keyword_path)
            return

        # porcupine = pvporcupine.create(access_key=self.porcupine_api_key, keyword_paths=[keyword_path], sensitivities=[1.0])
        porcupine = pvporcupine.create(access_key=self.porcupine_api_key, keywords=["computer"], sensitivities=[1.0])
        stream = self.p.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)
        
        print("\n\nListening for wake word...")
        try:
            while True:
                data = stream.read(porcupine.frame_length)
                pcm = np.frombuffer(data, dtype=np.int16)
                if porcupine.process(pcm) >= 0:
                    print("Wake word detected")
                    return True  # Return True if wake word is detected
        finally:
            # Properly stop and close the stream
            stream.stop_stream()
            stream.close()
            # Reinitialize PyAudio instance to avoid resource locking
            self.p.terminate()
            self.p = pyaudio.PyAudio()

        return False  # Return False if wake word is not detected

    def process_audio(self):
        # Record audio
        audio_path = self.record_audio_cobra()  # Assuming you use Cobra for recording

        # Transcribe audio
        transcription = self.transcribe_whisper(audio_path)  # Assuming you use Whisper for transcription

        # Optionally, you can perform additional actions based on the transcription here
        
        return transcription

    def run(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.listen_for_wake_word)
            wake_word_detected = future.result()

            if wake_word_detected:
                print("Initiating recording and transcription...")
                future = executor.submit(self.process_audio)
                transcription = future.result()
                return transcription
        return None  # If wake word is not detected, return None

            
if __name__ == "__main__":
    stt = STT()
    result = stt.run()
    print(f"\n\n>>>YOU SAID: {result}")
    

