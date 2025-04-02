#!/usr/bin/env python3

import os
import sys
import threading
import tempfile
import time
import wave
from typing import Optional, Dict, Any, Callable

# Import necessary libraries
try:
    import numpy as np
    from transcribers import BaseTranscriber, TranscriberFactory
    from config import get_config
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    sys.exit(1)


class LiveTranscriber:
    """Class for real-time transcription of audio from system output and microphone.
    
    This class provides functionality to transcribe audio in real-time from
    the system's output device and microphone input.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 transcription_callback: Optional[Callable[[str], None]] = None):
        """Initialize the live transcriber.
        
        Args:
            config: Configuration dictionary for transcription settings
            transcription_callback: Callback function for transcription results
        """
        # Load configuration
        self.config_obj = get_config()
        self.config = config or self.config_obj.config

        
        # Set up the transcriber
        self.engine = self.config.get("engine", "vosk")
        self.transcriber = TranscriberFactory.create_transcriber(self.config)
        
        # Transcription settings
        self.sample_rate = 16000  # Standard for speech recognition
        self.channels = 1  # Mono for transcription
        self.buffer_duration = 3.0  # Seconds of audio to process at once
        self.buffer_overlap = 1.0  # Seconds of overlap between buffers
        
        # Callback function
        self.transcription_callback = transcription_callback
        
        # Transcription state
        self.is_transcribing = False
        self.transcription_thread = None
        self.audio_buffer = np.array([])
        self.last_transcription = ""
    
    def start_transcription(self):
        """Start the transcription process."""
        if self.is_transcribing:
            print("Already transcribing")
            return
        
        print("Starting transcription...")
        self.is_transcribing = True
        self.audio_buffer = np.array([])
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._transcribe_loop)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
    def stop_transcription(self):
        """Stop the transcription process."""
        if not self.is_transcribing:
            print("Not transcribing")
            return
        
        print("Stopping transcription...")
        self.is_transcribing = False
        
        # Wait for thread to finish
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=1.0)
    
    def add_audio_data(self, audio_data: np.ndarray):
        """Add audio data to the buffer for transcription.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Debug: Print audio data info occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:  # Only print every 100th call
            print(f"Received audio data: shape={audio_data.shape}, dtype={audio_data.dtype}")
        
        # Ensure audio data is mono for transcription
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            if self._debug_counter % 100 == 0:
                print(f"Converted to mono: shape={audio_data.shape}")
        
        # Append to buffer
        if len(self.audio_buffer) == 0:
            self.audio_buffer = audio_data
        else:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        
        # Debug: Print buffer size occasionally
        if self._debug_counter % 100 == 0:
            print(f"Buffer size after adding data: {len(self.audio_buffer)}")
    
    def _transcribe_loop(self):
        """Main transcription loop.
        
        This method runs in a separate thread and continuously processes
        audio data from the buffer for transcription.
        """
        buffer_size = int(self.sample_rate * self.buffer_duration)
        overlap_size = int(self.sample_rate * self.buffer_overlap)
        
        print(f"Starting transcription loop with buffer size {buffer_size} and overlap {overlap_size}")
        
        # For debug printing
        loop_counter = 0
        last_buffer_debug = time.time()
        
        while self.is_transcribing:
            loop_counter += 1
            current_time = time.time()
            
            # Check if we have enough audio data to process
            if len(self.audio_buffer) >= buffer_size:
                # Extract a chunk of audio for processing
                audio_chunk = self.audio_buffer[:buffer_size]
                
                # Debug: Print buffer info occasionally
                if loop_counter % 10 == 0:  # Every 10th iteration
                    print(f"Processing audio chunk: buffer size={len(self.audio_buffer)}, chunk size={len(audio_chunk)}")
                
                # Keep the overlapping part for the next chunk
                if len(self.audio_buffer) > overlap_size:
                    self.audio_buffer = self.audio_buffer[buffer_size - overlap_size:]
                else:
                    self.audio_buffer = np.array([])
                
                # Process the audio chunk
                transcription = self._transcribe_chunk(audio_chunk)
                
                # Call the callback with the result if provided
                if transcription and self.transcription_callback:
                    self.transcription_callback(transcription)
            else:
                # Debug: Print when waiting for more audio (but not too often)
                if current_time - last_buffer_debug > 5.0:  # Only print every 5 seconds
                    print(f"Waiting for more audio data: current buffer size={len(self.audio_buffer)}, need {buffer_size}")
                    last_buffer_debug = current_time
            
            # Sleep to prevent CPU overuse
            time.sleep(0.1)
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a chunk of audio data.
        
        Args:
            audio_chunk: Audio data chunk as numpy array
            
        Returns:
            Transcribed text
        """
        try:
            # Save the audio chunk to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav_path = temp_file.name
            
            # Scale float32 data to int16 for WAV file
            if audio_chunk.dtype == np.float32:
                audio_chunk = (audio_chunk * 32767).astype(np.int16)
            
            # Save as WAV
            with wave.open(temp_wav_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_chunk.tobytes())
            
            # Transcribe the audio
            transcription = self.transcriber.transcribe_wav(temp_wav_path)
            
            # Debug: Print transcription result only if we got something
            if transcription:
                print(f"Transcription result: '{transcription}'")
            
            # Clean up the temporary file
            os.remove(temp_wav_path)
            
            # Update last transcription
            if transcription:
                self.last_transcription = transcription
                # Debug: Callback execution
                print(f"Executing transcription callback with text: '{transcription}'")
            
            return transcription
        
        except Exception as e:
            print(f"Error transcribing audio chunk: {e}")
            return ""
    
    def get_last_transcription(self) -> str:
        """Get the last transcription result.
        
        Returns:
            Last transcription text
        """
        return self.last_transcription