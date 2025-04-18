import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any

# Import ffmpeg for audio processing
try:
    import ffmpeg
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    import sys
    sys.exit(1)


class BaseTranscriber(ABC):
    """Abstract base class for audio transcription.

    This class defines the common interface and functionality that all transcribers
    should implement. Specific transcription engines should inherit from this class
    and implement the abstract methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the base transcriber.

        Args:
            config: Configuration dictionary for the transcriber
        """
        self.config = config

    @abstractmethod
    def load_model(self):
        """Load the transcription model.

        This method should be implemented by subclasses to load their specific models.
        """
        pass

    @abstractmethod
    def transcribe_wav(self, wav_path: str) -> str:
        """Transcribe a WAV file.

        Args:
            wav_path: Path to the WAV file

        Returns:
            Transcribed text
        """
        pass

    def transcribe_file_with_timestamps(self, input_path: str):
        """Transcribe an audio file (MP4 or WAV) and return text with timestamps.

        Args:
            input_path: Path to the audio file

        Returns:
            Tuple of (transcribed_text, segments)
            where segments is a list of dicts with keys: text, start, end
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist.")
            return "", []

        # Get the file extension
        file_ext = os.path.splitext(input_path)[1].lower()

        # Process based on file type
        try:
            if file_ext == ".mp4":
                # Extract audio from MP4
                wav_path = self.extract_audio_from_mp4(input_path)
                # Transcribe the extracted audio with timestamps
                text, segments = self.transcribe_wav_with_timestamps(wav_path)
                # Clean up the temporary file
                os.remove(wav_path)
                return text, segments
            elif file_ext == ".wav":
                # Transcribe the WAV file directly with timestamps
                return self.transcribe_wav_with_timestamps(input_path)
            else:
                print(f"Error: Unsupported file format: {file_ext}")
                print("Supported formats: .mp4, .wav")
                return "", []
        except Exception as e:
            print(f"Error during transcription with timestamps: {e}")
            return "", []

    def extract_audio_from_mp4(self, mp4_path: str, output_wav_path: Optional[str] = None) -> str:
        """Extract audio from an MP4 file and save as WAV.

        Args:
            mp4_path: Path to the MP4 file
            output_wav_path: Path to save the WAV file. If None, a temporary file is created.

        Returns:
            Path to the extracted WAV file
        """
        if output_wav_path is None:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_wav_path = temp_file.name
            temp_file.close()

        try:
            # Extract audio using ffmpeg
            (ffmpeg
                .input(mp4_path)
                .output(output_wav_path, acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
             )
            print(f"Audio extracted from {mp4_path} to {output_wav_path}")
            return output_wav_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e.stderr.decode()}")
            if os.path.exists(output_wav_path):
                os.remove(output_wav_path)
            raise

    def transcribe_file(self, input_path: str) -> str:
        """Transcribe an audio file (MP4 or WAV).

        Args:
            input_path: Path to the audio file

        Returns:
            Transcribed text
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist.")
            return ""

        # Get the file extension
        file_ext = os.path.splitext(input_path)[1].lower()

        # Process based on file type
        try:
            if file_ext == ".mp4":
                # Extract audio from MP4
                wav_path = self.extract_audio_from_mp4(input_path)
                # Transcribe the extracted audio
                text = self.transcribe_wav(wav_path)
                # Clean up the temporary file
                os.remove(wav_path)
                return text
            elif file_ext == ".wav":
                # Transcribe the WAV file directly
                return self.transcribe_wav(input_path)
            else:
                print(f"Error: Unsupported file format: {file_ext}")
                print("Supported formats: .mp4, .wav")
                return ""
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
