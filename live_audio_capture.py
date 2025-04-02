import sounddevice as sd
import wave
import tempfile
import os
import queue
import threading
import time
import numpy as np

class AudioRecorder:
    def __init__(self, channels=1, rate=16000, record_seconds=5):
        self.channels = channels
        self.rate = rate
        self.record_seconds = record_seconds
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Get default device info
        device_info = sd.query_devices(None, 'input')
        self.device = device_info['index']
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        
    def _record_audio(self):
        """Internal method to record audio in chunks"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f'Error: {status}')
            if self.is_recording:
                self.audio_queue.put(indata.copy())
        
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.rate,
            callback=audio_callback,
            blocksize=int(self.rate * self.record_seconds),
            device=self.device
        ):
            while self.is_recording:
                sd.sleep(100)
        
    def get_audio(self):
        """Get recorded audio from the queue"""
        if not self.audio_queue.empty():
            frames = self.audio_queue.get()
            # Check if frames contain any data
            return frames if frames.size > 0 else None
        return None
        
    def save_audio_to_file(self, frames, filename=None):
        """Save audio frames to a WAV file"""
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                filename = temp_audio.name
        
        frames = np.array(frames)
        if frames.ndim == 1:
            frames = frames.reshape(-1, 1)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.rate)
            wf.writeframes((frames * 32767).astype(np.int16).tobytes())
        
        return filename