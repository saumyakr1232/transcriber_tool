import os
import queue
import threading
import tempfile
import numpy as np
import soundcard as sc
import soundfile as sf

class AudioRecorder2:
    def __init__(self, channels=1, rate=16000, record_seconds=5, mic_device=None, system_device=None):
        self.channels = channels
        self.rate = rate
        self.record_seconds = record_seconds
        self.is_recording = False
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        # Get available devices
        self.available_mics = sc.all_microphones()
        self.available_system_devices = sc.all_microphones(include_loopback=True)
        
        # Set devices
        self.mic_device = mic_device if mic_device else sc.default_microphone()
        self.system_device = system_device if system_device else self._find_system_output_device()
        
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
        devices = self.available_system_devices
        return devices[0] if devices else None
        
    def get_available_mics(self):
        """Get list of available microphone devices"""
        return self.available_mics
        
    def get_available_system_devices(self):
        """Get list of available system audio devices"""
        return self.available_system_devices
        
    def set_mic_device(self, device):
        """Set microphone device"""
        self.mic_device = device
        
    def set_system_device(self, device):
        """Set system audio device"""
        self.system_device = device


    def _record_mic_audio(self):
        """Record audio from microphone"""
        block_size = int(self.rate * self.record_seconds)
        
        with self.mic_device.recorder(samplerate=self.rate, channels=self.channels, blocksize=50) as mic:
            while self.is_recording:
                data = mic.record(numframes=block_size)
                if data.size > 0:
                    self.mic_queue.put(data)
    
    def _record_system_audio(self):
        """Record audio from system output"""
        if self.system_device is None:
            return
            
        block_size = int(self.rate * self.record_seconds)
        
        with self.system_device.recorder(samplerate=self.rate, channels=self.channels, blocksize=50) as system:
            while self.is_recording:
                data = system.record(numframes=block_size)
                if data.size > 0:
                    self.system_queue.put(data)
        
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
        
        sf.write(filename, frames, self.rate)
        return filename