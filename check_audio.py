import wave
import sys

def check_audio_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        duration = wf.getnframes() / wf.getframerate()
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        
        print(f"File: {file_path}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Channels: {channels}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Frames: {frames}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_audio_file(sys.argv[1])
    else:
        print("Usage: python check_audio.py <audio_file.wav>")