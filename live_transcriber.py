import os
import sys
import threading
import tempfile
import time
import wave
import queue
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
                 transcription_callback: Optional[Callable[[str], None]] = None,
                 transcriber: Optional[BaseTranscriber] = None,
                 model_path: str = None):
        """Initialize the live transcriber.

        Args:
            config: Configuration dictionary for transcription settings
            transcription_callback: Callback function for transcription results
            model_path: Path to the model directory or model size
        """
        # Load configuration
        self.config_obj = get_config()
        self.config = config or self.config_obj.config

        # Set up the transcriber
        try:
            if not transcriber:
                self.transcriber = TranscriberFactory.create_transcriber(self.config, model_path)
            else:
                self.transcriber = transcriber
            self.engine = self.config.get("engine", "vosk")
            print(f"Using {self.engine} engine for live transcription.")
        except Exception as e:
            print(f"Error initializing transcriber: {e}")
            sys.exit(1)

        # Audio parameters
        self.chunk = 1024
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 5  # Process audio in 5-second chunks

        # Callback function
        self.transcription_callback = transcription_callback

        # Transcription state
        self.is_transcribing = False
        self.transcription_thread = None
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.last_transcription = ""
        
        # Silence detection parameters
        self.silence_threshold = self.config.get("silence_threshold", 0.05)  # Default threshold
        self.repetitive_patterns = [
            "a little bit of a little bit",
            "little bit of a little",
            "bit of a little bit",
            "of a little bit of a"
        ]

    def start_transcription(self):
        """Start the transcription process."""
        if self.is_transcribing:
            print("Already transcribing")
            return

        print("Starting transcription...")
        self.is_transcribing = True

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

    def add_audio_data(self, audio_data):
        """Add audio data to the buffer for transcription.

        Args:
            audio_data: Audio data as numpy array
        """
        if audio_data is None or (isinstance(audio_data, np.ndarray) and len(audio_data) == 0):
            print("No audio data provided")
            return

        self.audio_queue.put(audio_data)

    def _transcribe_loop(self):
        """Main transcription loop.

        This method runs in a separate thread and continuously processes
        audio data from the buffer for transcription.
        """
        while self.is_transcribing or not self.audio_queue.empty():
            if not self.audio_queue.empty():
                frames = self.audio_queue.get()
                # Process the audio chunk
                transcription = self._transcribe_chunk(frames)

                # If we got a transcription and have a callback, call it
                if transcription and self.transcription_callback:
                    self.transcription_callback(transcription)

            # Sleep to prevent CPU overuse
            time.sleep(0.1)

    def _transcribe_chunk(self, audio_chunk) -> str:
        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav_path = temp_file.name

            # Ensure audio data is in the correct format
            if isinstance(audio_chunk, list):
                # Join list of audio frames
                audio_data = b''.join(audio_chunk)
            elif isinstance(audio_chunk, np.ndarray):
                # Convert numpy array to bytes
                if audio_chunk.dtype != np.int16:
                    audio_chunk = (audio_chunk * 32767).astype(np.int16)
                audio_data = audio_chunk.tobytes()
                
                # Check for silence/background noise
                # Calculate RMS (Root Mean Square) as a measure of audio energy
                rms = np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32))))
                
                # Use the configured silence threshold
                if rms < self.silence_threshold:
                    # Audio is likely silence or background noise
                    # Skip transcription to avoid repetitive text
                    return ""
            else:
                # Assume it's already bytes
                audio_data = audio_chunk

            # Save as WAV with proper audio parameters
            with wave.open(temp_wav_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)

            # Transcribe the audio with timestamps
            transcription, segments = self.transcriber.transcribe_file_with_timestamps(temp_wav_path)

            # Clean up the temporary file
            try:
                os.remove(temp_wav_path)
            except:
                pass

            # Check for repetitive patterns that often occur during silence
            # Filter out transcriptions with repetitive patterns
            if transcription:
                is_repetitive = any(pattern in transcription.lower() for pattern in self.repetitive_patterns)
                if is_repetitive:
                    return ""
                    
                # Update last transcription
                self.last_transcription = transcription

                # If we have segments with timestamps, use the first one
                timestamp = None
                if segments and len(segments) > 0:
                    timestamp = segments[0].get("start", None)

                # Call the callback with timestamp if available
                if self.transcription_callback:
                    self.transcription_callback(transcription, timestamp)
                else:
                    self.transcription_queue.put(transcription)

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

    def get_next_transcription(self) -> Optional[str]:
        """Get the next transcription from the queue if available.

        Returns:
            Next transcription text or None if queue is empty
        """
        if not self.transcription_queue.empty():
            return self.transcription_queue.get()
        return None

    def load_model(self):
        """Load the transcription model.

        This method ensures the transcription model is properly initialized
        before use. It's called during application startup.
        """
        # The model is already loaded in the constructor
        # This method exists for compatibility with the application's initialization flow
        self.transcriber.load_model()
