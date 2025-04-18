import os
import sys
import tempfile
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any

# Import pyannote.audio for speaker diarization
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with 'pip install pyannote.audio' to use speaker diarization.")

# Import whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with 'pip install openai-whisper' to use Whisper engine.")


class SpeakerDiarizer:
    """Class for speaker diarization using pyannote.audio.

    This class provides functionality to identify different speakers in an audio file
    and segment the audio accordingly for transcription.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the speaker diarizer.

        Args:
            config: Configuration dictionary for the diarizer
        """
        self.config = config
        self.pipeline = None

        if not PYANNOTE_AVAILABLE:
            print("Error: Speaker diarization requires pyannote.audio.")
            print("Please install with: pip install pyannote.audio")
            return

        # Check if HuggingFace token is provided
        self.hf_token = config.get("diarization.hf_token", None)
        if not self.hf_token:
            print("Warning: No HuggingFace token provided for speaker diarization.")
            print("You may need to provide a token to use the pyannote.audio models.")
            print("Set it in the config under 'diarization.hf_token'.")

        # Load the diarization pipeline
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the pyannote.audio diarization pipeline."""
        try:
            # Use local model if specified, otherwise use the default model from HuggingFace
            model_path = self.config.get("diarization.model_path", None)

            if model_path and os.path.exists(model_path):
                print(f"Loading local diarization model from {model_path}...")
                self.pipeline = Pipeline.from_pretrained(model_path)
            else:
                print("Loading diarization model from HuggingFace...")
                # Default to the pyannote/speaker-diarization-3.1 model
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )

            print("Speaker diarization model loaded successfully.")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            self.pipeline = None

    def diarize(self, audio_path: str) -> Optional[Annotation]:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Annotation object containing speaker segments
        """
        if not self.pipeline:
            print("Error: Diarization pipeline not initialized.")
            return None

        try:
            # Apply the diarization pipeline
            diarization = self.pipeline(audio_path)
            return diarization
        except Exception as e:
            print(f"Error during diarization: {e}")
            return None

    def get_speaker_segments(self, audio_path: str) -> List[Dict[str, Any]]:
        """Get speaker segments with timestamps.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of dictionaries with speaker segments
            Each dictionary contains: {'speaker': speaker_id, 'start': start_time, 'end': end_time}
        """
        diarization = self.diarize(audio_path)
        if not diarization:
            return []

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            }
            segments.append(segment)

        return segments

    def transcribe_with_speakers(self, audio_path: str, whisper_model=None) -> List[Dict[str, Any]]:
        """Transcribe audio with speaker identification.

        Args:
            audio_path: Path to the audio file
            whisper_model: Optional pre-loaded whisper model

        Returns:
            List of dictionaries with transcribed segments
            Each dictionary contains: {'speaker': speaker_id, 'start': start_time, 'end': end_time, 'text': transcribed_text}
        """
        if not WHISPER_AVAILABLE:
            print("Error: Transcription requires Whisper.")
            return []

        # Get speaker segments
        speaker_segments = self.get_speaker_segments(audio_path)
        if not speaker_segments:
            return []

        # Load Whisper model if not provided
        if whisper_model is None:
            model_size = self.config.get("models.whisper.model_size", "base")
            device = "cuda" if torch.cuda.is_available() and self.config.get("models.whisper.use_gpu", True) else "cpu"
            whisper_model = whisper.load_model(model_size, device=device)

        # Create a temporary directory for audio segments
        with tempfile.TemporaryDirectory() as temp_dir:
            transcribed_segments = []

            # Process each speaker segment
            for i, segment in enumerate(speaker_segments):
                # Extract segment time range
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment['speaker']

                # Create segment file path
                segment_path = os.path.join(temp_dir, f"segment_{i}.wav")

                try:
                    # Extract segment using ffmpeg
                    import ffmpeg
                    (ffmpeg
                        .input(audio_path, ss=start_time, to=end_time)
                        .output(segment_path, acodec='pcm_s16le', ac=1, ar='16k')
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                     )

                    # Transcribe the segment
                    result = whisper_model.transcribe(segment_path)

                    # Add to transcribed segments
                    transcribed_segments.append({
                        'speaker': speaker,
                        'start': start_time,
                        'end': end_time,
                        'text': result['text'].strip()
                    })

                except Exception as e:
                    print(f"Error processing segment {i}: {e}")

            return transcribed_segments

    def format_transcript_with_speakers(self, segments: List[Dict[str, Any]]) -> str:
        """Format transcribed segments with speaker labels.

        Args:
            segments: List of transcribed segments with speaker information

        Returns:
            Formatted transcript with speaker labels and timestamps
        """
        if not segments:
            return ""

        transcript = ""
        for segment in segments:
            speaker = segment['speaker']
            start = self._format_time(segment['start'])
            end = self._format_time(segment['end'])
            text = segment['text']

            transcript += f"[{speaker}] {start} - {end}: {text}\n\n"

        return transcript

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to HH:MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    # Sample usage
    from config import get_config

    # Get configuration
    config = get_config().config

    # Create diarizer
    diarizer = SpeakerDiarizer(config)

    # Test with a sample file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Processing {audio_path}...")

        # Get speaker segments
        segments = diarizer.transcribe_with_speakers(audio_path)

        # Format transcript
        transcript = diarizer.format_transcript_with_speakers(segments)

        # Print transcript
        print("\nTranscript:")
        print(transcript)
    else:
        print("Usage: python speaker_diarization.py <audio_file>")
