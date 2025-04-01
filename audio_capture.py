#!/usr/bin/env python3

import os
import sys
import time
import wave
import threading
import tempfile
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Import necessary libraries
try:
    import numpy as np
    import sounddevice as sd
    from pydub import AudioSegment
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Import config module if available
try:
    from config import get_config
except ImportError:
    # Define a simple config if not available
    def get_config():
        return type('obj', (object,), {
            'get': lambda self, key, default=None: default,
            'config': {}
        })()


class AudioCapture:
    """Class for capturing audio from system output and microphone simultaneously.
    
    This class provides functionality to record audio from the system's default output
    device (what you hear) and the default microphone input simultaneously, and then
    merge them into a single WAV file.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the audio capture with configuration.
        
        Args:
            config: Configuration dictionary for audio capture settings
        """
        # Load configuration
        self.config_obj = get_config()
        self.config = config or {}
        
        # Set default parameters
        self.sample_rate = self.config.get("sample_rate", 44100)
        self.channels = self.config.get("channels", 2)  # Stereo
        self.dtype = self.config.get("dtype", 'float32')
        
        # Recording state
        self.is_recording = False
        self.system_audio_data = []
        self.mic_audio_data = []
        self.system_thread = None
        self.mic_thread = None
        
        # Get available devices
        self.devices = sd.query_devices()
        self.system_output_device = self._find_system_output_device()
        self.mic_input_device = self._find_mic_input_device()
        
        if self.system_output_device is None:
            print("Warning: Could not find system output device. System audio capture may not work.")
        
        if self.mic_input_device is None:
            print("Warning: Could not find microphone input device. Microphone capture may not work.")
    
    def _find_system_output_device(self) -> Optional[Dict]:
        """Find the system output device for recording.
        
        On macOS, this requires a virtual audio device like BlackHole or Soundflower
        to be installed and configured as the audio output.
        
        Returns:
            Dictionary with device info of the system output device, or None if not found
        """
        # Look for virtual audio devices first (BlackHole, Soundflower, etc.)
        virtual_device_keywords = ['blackhole', 'soundflower', 'loopback', 'virtual']
        
        for device in self.devices:
            # Check if this is a virtual audio device by name
            device_name = device['name'].lower()
            is_virtual = any(keyword in device_name for keyword in virtual_device_keywords)
            
            # Virtual devices should have input channels to capture system audio
            if is_virtual and device['max_input_channels'] > 0:
                print(f"Found virtual audio device: {device['name']}")
                return {
                    'index': device['index'],
                    'channels': min(2, device['max_input_channels'])
                }
        
        # If no virtual device found, try to use the default output device
        # (this won't work on macOS without additional software)
        try:
            device_info = sd.query_devices(kind='output')
            print("Warning: No virtual audio device found. System audio capture may not work.")
            print("On macOS, you need to install BlackHole or Soundflower to capture system audio.")
            print("See README_audio_capture.md for instructions.")
            
            # Check if the device has input channels (unlikely for regular output devices)
            if device_info['max_input_channels'] > 0:
                return {
                    'index': device_info['index'],
                    'channels': min(2, device_info['max_input_channels'])
                }
            else:
                print(f"Output device {device_info['name']} has no input channels and cannot be used for recording.")
                return None
                
        except Exception as e:
            print(f"Error finding system output device: {e}")
            return None
    
    def _find_mic_input_device(self) -> Optional[Dict]:
        """Find the default microphone input device.
        
        Returns:
            Dictionary with device info of the default microphone input device, or None if not found
        """
        try:
            # Try to get the default input device
            device_info = sd.query_devices(kind='input')
            return {
                'index': device_info['index'],
                'channels': min(2, device_info['max_input_channels'])
            }
        except Exception as e:
            print(f"Error finding microphone input device: {e}")
            
            # Try to find any input device
            for device in self.devices:
                if device['max_input_channels'] > 0:
                    print(f"Using alternative input device: {device['name']}")
                    return {
                        'index': device['index'],
                        'channels': min(2, device['max_input_channels'])
                    }
            
            return None
    
    def _record_system_audio(self):
        """Record audio from the system output device.
        
        On macOS, this requires a virtual audio device like BlackHole or Soundflower
        to be installed and configured as the audio output.
        """
        try:
            # Calculate buffer size based on sample rate to ensure accurate timing
            # Using 0.05 seconds of audio per buffer for better timing accuracy
            # Smaller buffer size helps prevent timing issues
            buffer_size = int(self.sample_rate * 0.05)  # 0.05 second buffer
            
            # Use blocksize parameter to ensure consistent timing
            with sd.InputStream(device=self.system_output_device['index'],
                            samplerate=self.sample_rate,
                            channels=self.system_output_device['channels'],
                            blocksize=buffer_size,
                            dtype=self.dtype) as stream:
                while self.is_recording:
                    data, overflowed = stream.read(buffer_size)
                    if overflowed:
                        print("System audio buffer overflowed")
                    self.system_audio_data.append(data.copy())
                    # Small sleep to prevent CPU overuse and ensure timing accuracy
                    time.sleep(0.001)
        except Exception as e:
            print(f"Error recording system audio: {e}")
            print("On macOS, you need to install a virtual audio device like BlackHole or Soundflower")
            print("and configure your system to route audio through it.")
            print("See README_audio_capture.md for instructions.")
            # Set empty data so the recording can continue with just microphone
            self.system_audio_data = []
    
    def _record_mic_audio(self):
        """Record audio from the microphone input device."""
        # Calculate buffer size based on sample rate to ensure accurate timing
        # Using 0.05 seconds of audio per buffer for better timing accuracy
        # Smaller buffer size helps prevent timing issues
        buffer_size = int(self.sample_rate * 0.05)  # 0.05 second buffer
        
        # Use blocksize parameter to ensure consistent timing
        with sd.InputStream(device=self.mic_input_device['index'],
                           samplerate=self.sample_rate,
                           channels=self.mic_input_device['channels'],
                           blocksize=buffer_size,
                           dtype=self.dtype) as stream:
            while self.is_recording:
                data, overflowed = stream.read(buffer_size)
                if overflowed:
                    print("Microphone audio buffer overflowed")
                self.mic_audio_data.append(data.copy())
                # Small sleep to prevent CPU overuse and ensure timing accuracy
                time.sleep(0.001)
    
    def start_recording(self):
        """Start recording audio from both system output and microphone."""
        if self.is_recording:
            print("Already recording")
            return
        
        print("Starting audio capture...")
        self.is_recording = True
        self.system_audio_data = []
        self.mic_audio_data = []
        
        # Start system audio recording thread if device available
        if self.system_output_device is not None:
            self.system_thread = threading.Thread(target=self._record_system_audio)
            self.system_thread.daemon = True
            self.system_thread.start()
            print(f"Recording system audio from device {self.system_output_device['index']} with {self.system_output_device['channels']} channels")
        
        # Start microphone recording thread if device available
        if self.mic_input_device is not None:
            self.mic_thread = threading.Thread(target=self._record_mic_audio)
            self.mic_thread.daemon = True
            self.mic_thread.start()
            print(f"Recording microphone audio from device {self.mic_input_device['index']} with {self.mic_input_device['channels']} channels")
    
    def stop_recording(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stop recording and return the captured audio data.
        
        Returns:
            Tuple of (system_audio, mic_audio) as numpy arrays
        """
        if not self.is_recording:
            print("Not recording")
            return np.array([]), np.array([])
        
        print("Stopping audio capture...")
        self.is_recording = False
        
        # Wait for threads to finish
        if self.system_thread and self.system_thread.is_alive():
            self.system_thread.join(timeout=1.0)
        
        if self.mic_thread and self.mic_thread.is_alive():
            self.mic_thread.join(timeout=1.0)
        
        # Concatenate audio data
        system_audio = np.vstack(self.system_audio_data) if self.system_audio_data else np.array([])
        mic_audio = np.vstack(self.mic_audio_data) if self.mic_audio_data else np.array([])
        
        # Calculate and print actual durations
        system_duration = len(system_audio) / self.sample_rate if len(system_audio) > 0 else 0
        mic_duration = len(mic_audio) / self.sample_rate if len(mic_audio) > 0 else 0
        
        print(f"Captured {system_duration:.2f}s of system audio ({len(self.system_audio_data)} buffers)")
        print(f"Captured {mic_duration:.2f}s of microphone audio ({len(self.mic_audio_data)} buffers)")
        
        # Debug information about sample rate and buffer sizes
        print(f"Sample rate: {self.sample_rate} Hz")
        if self.system_audio_data and len(self.system_audio_data) > 0:
            print(f"Average system buffer size: {len(system_audio) / len(self.system_audio_data):.1f} frames")
        if self.mic_audio_data and len(self.mic_audio_data) > 0:
            print(f"Average mic buffer size: {len(mic_audio) / len(self.mic_audio_data):.1f} frames")
        
        return system_audio, mic_audio
    
    def _save_audio_to_wav(self, audio_data: np.ndarray, file_path: str):
        """Save audio data to a WAV file.
        
        Args:
            audio_data: Audio data as numpy array
            file_path: Path to save the WAV file
        """
        if len(audio_data) == 0:
            print(f"No audio data to save to {file_path}")
            return
        
        # Scale float32 data to int16 for WAV file
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate * 0.5)
            wf.writeframes(audio_data.tobytes())
        
        print(f"Audio saved to {file_path}")
    
    def merge_audio(self, system_audio: np.ndarray, mic_audio: np.ndarray) -> np.ndarray:
        """Merge system audio and microphone audio.
        
        Args:
            system_audio: System audio data as numpy array
            mic_audio: Microphone audio data as numpy array
            
        Returns:
            Merged audio data as numpy array
        """
        # Handle case where one of the sources has no data
        if len(system_audio) == 0:
            return mic_audio
        if len(mic_audio) == 0:
            return system_audio
        
        # Get the number of channels for each audio source
        system_channels = system_audio.shape[1] if system_audio.ndim > 1 else 1
        mic_channels = mic_audio.shape[1] if mic_audio.ndim > 1 else 1
        
        # Convert mono to stereo if needed for mixing
        output_channels = max(system_channels, mic_channels)
        
        # Convert mono to matching channel count if needed
        if system_audio.ndim == 1 or system_channels < output_channels:
            system_audio = np.repeat(system_audio.reshape(-1, 1), output_channels, axis=1)
        if mic_audio.ndim == 1 or mic_channels < output_channels:
            mic_audio = np.repeat(mic_audio.reshape(-1, 1), output_channels, axis=1)
        
        # Ensure both arrays have the same length
        max_length = max(len(system_audio), len(mic_audio))
        if len(system_audio) < max_length:
            padding = np.zeros((max_length - len(system_audio), output_channels), dtype=system_audio.dtype)
            system_audio = np.vstack([system_audio, padding])
        if len(mic_audio) < max_length:
            padding = np.zeros((max_length - len(mic_audio), output_channels), dtype=mic_audio.dtype)
            mic_audio = np.vstack([mic_audio, padding])
        
        # Mix the audio (equal weights for system and mic audio)
        # You can adjust the mixing ratio if needed
        system_weight = 0.5  # 50% system audio
        mic_weight = 0.5     # 50% microphone audio
        merged_audio = (system_audio * system_weight) + (mic_audio * mic_weight)
        
        return merged_audio
    
    def save_merged_audio(self, output_path: str) -> str:
        """Save the merged audio to a WAV file.
        
        Args:
            output_path: Path to save the merged WAV file
            
        Returns:
            Path to the saved WAV file
        """
        # Stop recording if still recording
        if self.is_recording:
            system_audio, mic_audio = self.stop_recording()
        else:
            system_audio = np.vstack(self.system_audio_data) if self.system_audio_data else np.array([])
            mic_audio = np.vstack(self.mic_audio_data) if self.mic_audio_data else np.array([])
        
        # Check if we have any audio data
        if len(system_audio) == 0 and len(mic_audio) == 0:
            print("No audio data to save")
            return ""
        
        # Merge the audio
        merged_audio = self.merge_audio(system_audio, mic_audio)
        
        # Save to file
        self._save_audio_to_wav(merged_audio, output_path)
        
        return output_path


def main():
    """Main function to run the audio capture from command line."""
    parser = argparse.ArgumentParser(description="Capture audio from system output and microphone.")
    parser.add_argument("--output", "-o", default="captured_audio.wav", 
                        help="Path to the output WAV file (default: captured_audio.wav)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                        help="Duration to record in seconds (default: 10.0)")
    parser.add_argument("--sample-rate", "-r", type=int, default=44100,
                        help="Sample rate in Hz (default: 44100)")
    parser.add_argument("--list-devices", "-l", action="store_true",
                        help="List available audio devices and exit")
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Create audio capture instance with custom sample rate
    config = {"sample_rate": args.sample_rate}
    audio_capture = AudioCapture(config)
    
    try:
        print(f"Recording for {args.duration} seconds...")
        # Record start time for precise duration calculation
        start_time = time.time()
        audio_capture.start_recording()
        
        # Wait for specified duration with more precise timing
        # Use a loop with small sleeps to ensure accurate duration
        elapsed = 0
        while elapsed < args.duration:
            # Sleep in small increments to allow for more precise timing
            time.sleep(0.05)
            elapsed = time.time() - start_time
            # Optionally print progress (uncomment if needed)
            # if int(elapsed) != int(elapsed - 0.05) and elapsed < args.duration:
            #     print(f"Recording: {elapsed:.1f}s / {args.duration:.1f}s")
        
        # Calculate actual recording duration
        actual_duration = time.time() - start_time
        print(f"Actual recording duration: {actual_duration:.2f}s")
        
        # Stop recording and save
        audio_capture.stop_recording()
        output_path = audio_capture.save_merged_audio(args.output)
        
        if output_path:
            print(f"\nAudio saved to: {output_path}")
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
        audio_capture.stop_recording()
        output_path = audio_capture.save_merged_audio(args.output)
        if output_path:
            print(f"\nAudio saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()