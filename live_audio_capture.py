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
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        # Get default input device info
        input_device_info = sd.query_devices(None, 'input')
        self.mic_device = input_device_info['index']
        
        # Find system output device (virtual audio device)
        self.system_device = self._find_system_output_device()
        
    def start_recording(self):
        """Start recording audio from both microphone and system output"""
        self.is_recording = True
        
        # Start microphone recording thread
        self.mic_thread = threading.Thread(target=self._record_mic_audio)
        self.mic_thread.daemon = True
        self.mic_thread.start()
        
        # Start system audio recording thread if device available
        if self.system_device is not None:
            self.system_thread = threading.Thread(target=self._record_system_audio)
            self.system_thread.daemon = True
            self.system_thread.start()
        else:
            print("Warning: No virtual audio device found. System audio capture may not work.")
            print("On macOS, you need to install BlackHole or Soundflower to capture system audio.")
        
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        
    def _find_system_output_device(self):
        """Find the system output device for recording."""
        virtual_device_keywords = ['blackhole', 'soundflower', 'loopback', 'virtual']
        
        devices = sd.query_devices()
        for device in devices:
            device_name = device['name'].lower()
            is_virtual = any(keyword in device_name for keyword in virtual_device_keywords)
            
            if is_virtual and device['max_input_channels'] > 0:
                print(f"Found virtual audio device: {device['name']}")
                return device['index']
        
        print("Warning: No virtual audio device found. System audio capture may not work.")
        print("On macOS, you need to install BlackHole or Soundflower to capture system audio.")
        return None

    def _record_mic_audio(self):
        """Record audio from microphone"""
        def mic_callback(indata, frames, time, status):
            if status:
                print(f'Mic Error: {status}')
            if self.is_recording:
                self.mic_queue.put(indata.copy())
        
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.rate,
            callback=mic_callback,
            blocksize=int(self.rate * self.record_seconds),
            device=self.mic_device
        ):
            while self.is_recording:
                sd.sleep(100)
    
    def _record_system_audio(self):
        """Record audio from system output"""
        if self.system_device is None:
            return
            
        def system_callback(indata, frames, time, status):
            if status:
                print(f'System Error: {status}')
            if self.is_recording:
                self.system_queue.put(indata.copy())
        
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.rate,
            callback=system_callback,
            blocksize=int(self.rate * self.record_seconds),
            device=self.system_device
        ):
            while self.is_recording:
                sd.sleep(100)
        
    def get_mic_audio(self):
        """Get recorded audio from the microphone queue"""
        if not self.mic_queue.empty():
            frames = self.mic_queue.get()
            return frames if frames.size > 0 else None
        return None
    
    def get_system_audio(self):
        """Get recorded audio from the system queue"""
        if not self.system_queue.empty():
            frames = self.system_queue.get()
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