import os
import sys
from typing import Optional, Dict, Any

# Import necessary libraries
try:
    from .base_transcriber import BaseTranscriber
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Try to import whisper if available
try:
    import whisper
    import torch  # Add this import
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with 'pip install openai-whisper' to use Whisper engine.")


class WhisperTranscriber(BaseTranscriber):
    """Transcriber implementation using the Whisper speech recognition engine.
    
    This class handles transcription of audio files using the OpenAI Whisper model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Whisper transcriber.
        
        Args:
            config: Configuration dictionary for the transcriber
        """
        super().__init__(config)
        
        if not WHISPER_AVAILABLE:
            print("Error: Whisper engine selected but not available.")
            print("Please install Whisper with: pip install openai-whisper")
            sys.exit(1)
            
        self.model = None
        self.model_size = None
        self.fp16 = False
    
    def load_model(self, model_size: str = None):
        """Load the Whisper model.
        
        Args:
            model_size: Size of the Whisper model to load. If None, will use the configured model size.
        """
        # Get model size from config if not specified
        if model_size is None:
            model_size = self.config.get("models.whisper.model_size", "base")
        
        # Check if model size is valid
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if model_size not in valid_sizes:
            print(f"Error: Invalid Whisper model size '{model_size}'")
            print(f"Valid sizes: {', '.join(valid_sizes)}")
            sys.exit(1)
        
        # Get device preference
        use_gpu = self.config.get("models.whisper.use_gpu", True)
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Load the model
        try:
            print(f"Loading Whisper model '{model_size}' on {device}...")
            self.model = whisper.load_model(model_size, device=device)
            # Set fp16 only if using GPU
            self.fp16 = device == "cuda"
            self.model_size = model_size
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            sys.exit(1)
    
    def transcribe_wav(self, wav_path: str) -> str:
        """Transcribe a WAV file using Whisper.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Transcribed text
        """
        try:
            # Convert path to pathlib.Path for cross-platform compatibility
            from pathlib import Path
            import os
            
            # Normalize the path for Windows compatibility
            wav_path = os.path.abspath(os.path.normpath(wav_path))
            wav_file = Path(wav_path)
            
            if not wav_file.exists():
                print(f"Error: WAV file not found at {wav_path}")
                return ""
            
            # Convert to absolute path and resolve any symlinks
            try:
                wav_file = wav_file.resolve(strict=True)
            except (RuntimeError, OSError) as e:
                print(f"Error: Unable to resolve WAV file path - {e}")
                return ""
            
            # Additional Windows-specific path check
            if os.name == 'nt' and len(str(wav_file)) > 260:
                print("Error: File path exceeds Windows path length limit")
                return ""
            
            # Get language preference from config
            language = self.config["models"]["whisper"]["language"]
            if language == "":
                language = None
            
            # Transcribe the audio using the resolved path as string
            print("Transcribing with Whisper... Language: ", language)
            result = self.model.transcribe(
                str(wav_file),
                language=language,
                verbose=False,
                fp16=self.fp16
            )
            
            # Return the transcribed text
            return result["text"].strip()
        
        except FileNotFoundError as e:
            print(f"Error: Could not access WAV file - {e}")
            return ""
        except PermissionError as e:
            print(f"Error: Permission denied accessing WAV file - {e}")
            return ""
        except Exception as e:
            print(f"Error transcribing audio with Whisper: {e}")
            return ""