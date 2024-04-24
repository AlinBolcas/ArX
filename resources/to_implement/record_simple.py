import pyaudio
from pydub import AudioSegment

def record_audio(record_seconds, filename="output"):
    # Audio settings
    chunk = 1024  # Record in chunks
    format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Mono
    sample_rate = 44100  # Sample rate

    p = pyaudio.PyAudio()

    # Open stream for recording
    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")

    frames = []

    # Record data
    for _ in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert audio data to MP3 format without saving an intermediate WAV file
    audio_data = b''.join(frames)
    audio = AudioSegment(
        audio_data, frame_rate=sample_rate, sample_width=2, channels=1
    )
    audio.export(filename + '.mp3', format="mp3")

    print("Recording saved as " + filename + ".mp3")

# Example usage
record_audio(5, "test_audio")
