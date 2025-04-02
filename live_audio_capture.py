#!/usr/bin/env python3

import os
import sys
import time
import wave
import threading
import tempfile
import queue
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable

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


class LiveAudioCapture:
    """Class for capturing and streaming audio from system output and microphone simultaneously.
    
    This class extends the functionality of AudioCapture to provide real-time streaming
    of audio data for visualization and transcription.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 system_callback: Optional[Callable[[np.ndarray], None]] = None,
                 mic_callback: Optional[Callable[[np.ndarray], None]] = None,
                 merged_callback: Optional[Callable[[np.ndarray], None]] = None,
                 buffer_size_seconds: float = 0.05):
        """Initialize the live audio capture with configuration.
        
        Args:
            config: Configuration dictionary for audio capture settings
            system_callback: Callback function for system audio data
            mic_callback: Callback function for microphone audio data
            merged_callback: Callback function for merged audio data
            buffer_size_seconds: Size of audio buffer in seconds
        """
        # Load configuration
        self.config_obj = get_config()
        self.config = config or {}
        
        # Set default parameters
        self.sample_rate = self.config.get("sample_rate", 16000)  # Lower sample rate for transcription
        self.channels = self.config.get("channels", 1)  # Mono for transcription
        self.dtype = self.config.get("dtype", 'float32')
        
        # Callback functions
        self.system_callback = system_callback
        self.mic_callback = mic_callback
        self.merged_callback = merged_callback
        
        # Buffer size for audio processing
        self.buffer_size_seconds = buffer_size_seconds
        self.buffer_size = int(self.sample_rate * buffer_size_seconds)
        
        # Recording state
        self.is_recording = False
        self.is_transcribing = False
        self.system_thread = None
        self.mic_thread = None
        self.processing_thread = None
        
        # Audio data queues for processing
        self.system_queue = queue.Queue()
        self.mic_queue = queue.Queue()
        
        # For visualization: keep a short history of audio data
        self.viz_buffer_size = 1024  # Number of samples to keep for visualization
        self.system_viz_buffer = np.zeros(self.viz_buffer_size)
        self.mic_viz_buffer = np.zeros(self.viz_buffer_size)
        self.merged_viz_buffer = np.zeros(self.viz_buffer_size)
        
        # For saving: accumulate all audio data
        self.system_audio_data = []
        self.mic_audio_data = []
        
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
                    'channels': min(self.channels, device['max_input_channels'])
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
                    'channels': min(self.channels, device_info['max_input_channels'])
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
                'channels': min(self.channels, device_info['max_input_channels'])
            }
        except Exception as e:
            print(f"Error finding microphone input device: {e}")
            
            # Try to find any input device
            for device in self.devices:
                if device['max_input_channels'] > 0:
                    print(f"Using alternative input device: {device['name']}")
                    return {
                        'index': device['index'],
                        'channels': min(self.channels, device['max_input_channels'])
                    }
            
            return None
    
    def _stream_system_audio(self):
        """Stream audio from the system output device.
        
        This method continuously captures audio from the system output device
        and adds it to the processing queue.
        """
        try:
            with sd.InputStream(device=self.system_output_device['index'],
                            samplerate=self.sample_rate,
                            channels=self.system_output_device['channels'],
                            blocksize=self.buffer_size,
                            dtype=self.dtype) as stream:
                while self.is_recording:
                    data, overflowed = stream.read(self.buffer_size)
                    if overflowed:
                        print("System audio buffer overflowed")
                    
                    # Add to queue for processing
                    self.system_queue.put(data.copy())
                    
                    # Also save for later if needed
                    self.system_audio_data.append(data.copy())
                    
                    # Small sleep to prevent CPU overuse
                    time.sleep(0.001)
        except Exception as e:
            print(f"Error streaming system audio: {e}")
            print("On macOS, you need to install a virtual audio device like BlackHole or Soundflower")
            print("and configure your system to route audio through it.")
            print("See README_audio_capture.md for instructions.")
    
    def _stream_mic_audio(self):
        """Stream audio from the microphone input device.
        
        This method continuously captures audio from the microphone input device
        and adds it to the processing queue.
        """
        try:
            with sd.InputStream(device=self.mic_input_device['index'],
                            samplerate=self.sample_rate,
                            channels=self.mic_input_device['channels'],
                            blocksize=self.buffer_size,
                            dtype=self.dtype) as stream:
                while self.is_recording:
                    data, overflowed = stream.read(self.buffer_size)
                    if overflowed:
                        print("Microphone audio buffer overflowed")
                    
                    # Add to queue for processing
                    self.mic_queue.put(data.copy())
                    
                    # Also save for later if needed
                    self.mic_audio_data.append(data.copy())
                    
                    # Small sleep to prevent CPU overuse
                    time.sleep(0.001)
        except Exception as e:
            print(f"Error streaming microphone audio: {e}")
    
    def _process_audio(self):
        """Process audio data from the queues.
        
        This method continuously processes audio data from the system and microphone
        queues, merges them, and calls the appropriate callbacks.
        """
        while self.is_recording:
            # Process system audio if available
            if not self.system_queue.empty():
                try:
                    system_data = self.system_queue.get(block=False)
                    
                    # Update visualization buffer
                    self._update_viz_buffer(system_data, 'system')
                    
                    # Call system callback if provided
                    if self.system_callback:
                        self.system_callback(system_data)
                except queue.Empty:
                    pass
            
            # Process microphone audio if available
            if not self.mic_queue.empty():
                try:
                    mic_data = self.mic_queue.get(block=False)
                    
                    # Update visualization buffer
                    self._update_viz_buffer(mic_data, 'mic')
                    
                    # Call microphone callback if provided
                    if self.mic_callback:
                        self.mic_callback(mic_data)
                except queue.Empty:
                    pass
            
            # Process merged audio if both audio sources are available
            if (len(self.system_audio_data) > 0 and len(self.mic_audio_data) > 0 and 
                self.merged_callback is not None):
                # Get the latest chunks from available sources
                system_chunk = self.system_audio_data[-1]
                mic_chunk = self.mic_audio_data[-1]
                
                # Merge the audio
                merged_data = self._merge_audio_chunks(system_chunk, mic_chunk)
                
                # Update visualization buffer
                self._update_viz_buffer(merged_data, 'merged')
                
                # Call merged callback
                self.merged_callback(merged_data)
            # If only one audio source is available and merged callback exists, pass that source directly
            elif self.merged_callback is not None:
                if len(self.system_audio_data) > 0:
                    # Use system audio only
                    system_chunk = self.system_audio_data[-1]
                    self._update_viz_buffer(system_chunk, 'merged')
                    self.merged_callback(system_chunk)
                elif len(self.mic_audio_data) > 0:
                    # Use mic audio only
                    mic_chunk = self.mic_audio_data[-1]
                    self._update_viz_buffer(mic_chunk, 'merged')
                    self.merged_callback(mic_chunk)
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.01)
    
    def _update_viz_buffer(self, data: np.ndarray, source: str):
        """Update the visualization buffer with new audio data.
        
        Args:
            data: New audio data
            source: Source of the audio data ('system', 'mic', or 'merged')
        """
        # Ensure data is 1D for visualization
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        # Get the appropriate buffer
        if source == 'system':
            buffer = self.system_viz_buffer
        elif source == 'mic':
            buffer = self.mic_viz_buffer
        elif source == 'merged':
            buffer = self.merged_viz_buffer
        else:
            return
        
        # Shift the buffer and add new data
        if len(data) < len(buffer):
            # Shift buffer by the length of new data
            buffer[:-len(data)] = buffer[len(data):]
            # Add new data at the end
            buffer[-len(data):] = data
        else:
            # If new data is larger than buffer, just take the last part
            buffer[:] = data[-len(buffer):]
    
    def _merge_audio_chunks(self, system_chunk: np.ndarray, mic_chunk: np.ndarray) -> np.ndarray:
        """Merge system and microphone audio chunks.
        
        Args:
            system_chunk: System audio data chunk
            mic_chunk: Microphone audio data chunk
            
        Returns:
            Merged audio data
        """
        # Ensure both chunks have the same shape
        if system_chunk.shape != mic_chunk.shape:
            # Reshape to match the smaller one
            min_len = min(len(system_chunk), len(mic_chunk))
            system_chunk = system_chunk[:min_len]
            mic_chunk = mic_chunk[:min_len]
        
        # Mix the audio (equal weights)
        system_weight = 0.5
        mic_weight = 0.5
        merged_chunk = (system_chunk * system_weight) + (mic_chunk * mic_weight)
        
        return merged_chunk
    
    def get_visualization_data(self, source: str = 'merged') -> np.ndarray:
        """Get the current visualization data buffer.
        
        Args:
            source: Source of the audio data ('system', 'mic', or 'merged')
            
        Returns:
            Visualization data as numpy array
        """
        if source == 'system':
            return self.system_viz_buffer.copy()
        elif source == 'mic':
            return self.mic_viz_buffer.copy()
        else:  # Default to merged
            return self.merged_viz_buffer.copy()
    
    def start_recording(self):
        """Start recording and streaming audio from both system output and microphone."""
        if self.is_recording:
            print("Already recording")
            return
        
        print("Starting audio capture...")
        self.is_recording = True
        
        # Clear previous data
        self.system_audio_data = []
        self.mic_audio_data = []
        self.system_queue = queue.Queue()
        self.mic_queue = queue.Queue()
        
        # Start system audio streaming thread if device available
        if self.system_output_device is not None:
            self.system_thread = threading.Thread(target=self._stream_system_audio)
            self.system_thread.daemon = True
            self.system_thread.start()
            print(f"Recording system audio from device {self.system_output_device['index']} with {self.system_output_device['channels']} channels")
        
        # Start microphone streaming thread if device available
        if self.mic_input_device is not None:
            self.mic_thread = threading.Thread(target=self._stream_mic_audio)
            self.mic_thread.daemon = True
            self.mic_thread.start()
            print(f"Recording microphone audio from device {self.mic_input_device['index']} with {self.mic_input_device['channels']} channels")
        
        # Start audio processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
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
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Concatenate audio data
        system_audio = np.vstack(self.system_audio_data) if self.system_audio_data else np.array([])
        mic_audio = np.vstack(self.mic_audio_data) if self.mic_audio_data else np.array([])
        
        # Calculate and print actual durations
        system_duration = len(system_audio) / self.sample_rate if len(system_audio) > 0 else 0
        mic_duration = len(mic_audio) / self.sample_rate if len(mic_audio) > 0 else 0
        
        print(f"Captured {system_duration:.2f}s of system audio ({len(self.system_audio_data)} buffers)")
        print(f"Captured {mic_duration:.2f}s of microphone audio ({len(self.mic_audio_data)} buffers)")
        
        return system_audio, mic_audio
    
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
        merged_audio = self._merge_audio(system_audio, mic_audio)
        
        # Save to file
        self._save_audio_to_wav(merged_audio, output_path)
        
        return output_path
    
    def _merge_audio(self, system_audio: np.ndarray, mic_audio: np.ndarray) -> np.ndarray:
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
        system_weight = 0.5  # 50% system audio
        mic_weight = 0.5     # 50% microphone audio
        merged_audio = (system_audio * system_weight) + (mic_audio * mic_weight)
        
        return merged_audio
    
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
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        print(f"Audio saved to {file_path}")


def main():
    """Main function to test the live audio capture."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test live audio capture from system output and microphone.")
    parser.add_argument("--output", "-o", default="captured_audio.wav", 
                        help="Path to the output WAV file (default: captured_audio.wav)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                        help="Duration to record in seconds (default: 10.0)")
    parser.add_argument("--sample-rate", "-r", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
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
    
    # Define simple callbacks for testing
    def system_callback(data):
        rms = np.sqrt(np.mean(data**2))
        print(f"System audio level: {rms:.6f}", end="\r")
    
    def mic_callback(data):
        rms = np.sqrt(np.mean(data**2))
        print(f"Mic audio level: {rms:.6f}", end="\r")
    
    # Create the live audio capture instance
    audio_capture = LiveAudioCapture(
        config=config,
        system_callback=system_callback,
        mic_callback=mic_callback
    )
    
    try:
        print(f"Recording for {args.duration} seconds...")
        # Start recording
        audio_capture.start_recording()
        
        # Wait for specified duration
        time.sleep(args.duration)
        
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