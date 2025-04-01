#!/usr/bin/env python3

import os
import sys
import json
import argparse
import wave
import subprocess
from pathlib import Path
import tempfile
from typing import Optional, Union, Dict, Any

# Import necessary libraries
try:
    import numpy as np
    from vosk import Model, KaldiRecognizer
    from pydub import AudioSegment
    import ffmpeg
    # Import config module
    from config import get_config
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Try to import whisper if available
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with 'pip install openai-whisper' to use Whisper engine.")

class AudioTranscriber:
    """Class for transcribing audio from MP4 or WAV files using offline methods."""
    
    def __init__(self, model_path: str = None):
        """Initialize the transcriber with a model.
        
        Args:
            model_path: Path to the model directory. If None, will use the configured model.
        """
        # Load configuration
        self.config = get_config()
        self.engine = self.config.get_engine()
        
        # Initialize the appropriate engine
        if self.engine == "vosk":
            self._init_vosk(model_path)
        elif self.engine == "whisper":
            if not WHISPER_AVAILABLE:
                print("Error: Whisper engine selected but not available.")
                print("Please install Whisper with: pip install openai-whisper")
                sys.exit(1)
            self._init_whisper(model_path)
        else:
            print(f"Error: Unknown engine '{self.engine}'")
            print("Supported engines: 'vosk', 'whisper'")
            sys.exit(1)
    
    def _init_vosk(self, model_path: str = None):
        """Initialize the Vosk engine.
        
        Args:
            model_path: Path to the Vosk model directory. If None, will use the configured model.
        """
        # Try to find a model if not specified
        if model_path is None:
            model_path = self.config.get("models.vosk.model_path")
            
            # If still None, look in the models directory
            if model_path is None:
                models_dir = Path("./models")
                if models_dir.exists():
                    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("vosk-model")]
                    if model_dirs:
                        model_path = str(model_dirs[0])
                        print(f"Using model: {model_path}")
                    else:
                        print("No Vosk model found in ./models/ directory.")
                        print("Please download a model from https://alphacephei.com/vosk/models")
                        print("and extract it to the ./models/ directory.")
                        sys.exit(1)
                else:
                    print("Models directory not found.")
                    print("Please create a ./models/ directory and download a Vosk model")
                    print("from https://alphacephei.com/vosk/models")
                    sys.exit(1)
        
        # Load the model
        try:
            self.model = Model(model_path)
            print("Vosk model loaded successfully.")
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
            sys.exit(1)
    
    def _init_whisper(self, model_path: str = None):
        """Initialize the Whisper engine.
        
        Args:
            model_path: Path to the Whisper model or model size. If None, will use the configured model.
        """
        # Get model size from config if not specified
        if model_path is None:
            model_size = self.config.get("models.whisper.model_size", "base")
        else:
            model_size = model_path
        
        # Check if model size is valid
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if model_size not in valid_sizes:
            print(f"Error: Invalid Whisper model size '{model_size}'")
            print(f"Valid sizes: {', '.join(valid_sizes)}")
            sys.exit(1)
        
        # Get device preference
        use_gpu = self.config.get("models.whisper.use_gpu", True)
        device = "cuda" if use_gpu and whisper.available_devices() else "cpu"
        
        # Load the model
        try:
            print(f"Loading Whisper model '{model_size}' on {device}...")
            self.model = whisper.load_model(model_size, device=device)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            sys.exit(1)
    
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
            sys.exit(1)
    
    def transcribe_wav_vosk(self, wav_path: str) -> str:
        """Transcribe a WAV file using Vosk.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Transcribed text
        """
        try:
            # Open the WAV file
            wf = wave.open(wav_path, "rb")
            
            # Check if the WAV file is in the correct format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                print("Audio file must be WAV format mono PCM.")
                # Convert the WAV file to the correct format
                print("Converting audio to the correct format...")
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav_path = temp_file.name
                temp_file.close()
                
                (ffmpeg
                    .input(wav_path)
                    .output(temp_wav_path, acodec='pcm_s16le', ac=1, ar='16k')
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                
                wf.close()
                wf = wave.open(temp_wav_path, "rb")
            
            # Create a recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            # Process the audio
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.append(part_result)
            
            part_result = json.loads(rec.FinalResult())
            results.append(part_result)
            
            # Extract the text from the results
            text = ""
            for res in results:
                if "text" in res:
                    text += res["text"] + " "
            
            wf.close()
            return text.strip()
        
        except Exception as e:
            print(f"Error transcribing audio with Vosk: {e}")
            return ""
    
    def transcribe_wav_whisper(self, wav_path: str) -> str:
        """Transcribe a WAV file using Whisper.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Transcribed text
        """
        try:
            # Get language preference from config
            language = self.config.get("models.whisper.language", None)
            if language == "":
                language = None
            
            # Transcribe the audio
            print("Transcribing with Whisper...")
            result = self.model.transcribe(
                wav_path,
                language=language,
                verbose=False
            )
            
            # Return the transcribed text
            return result["text"].strip()
        
        except Exception as e:
            print(f"Error transcribing audio with Whisper: {e}")
            return ""
    
    def transcribe_wav(self, wav_path: str) -> str:
        """Transcribe a WAV file using the configured engine.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Transcribed text
        """
        if self.engine == "vosk":
            return self.transcribe_wav_vosk(wav_path)
        elif self.engine == "whisper":
            return self.transcribe_wav_whisper(wav_path)
        else:
            print(f"Error: Unknown engine '{self.engine}'")
            return ""
    
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


def main():
    """Main function to run the transcriber from command line."""
    parser = argparse.ArgumentParser(description="Transcribe audio from MP4 or WAV files.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input audio file (MP4 or WAV)")
    parser.add_argument("--output", "-o", help="Path to the output text file")
    parser.add_argument("--model", "-m", help="Path to the model directory or model size")
    parser.add_argument("--engine", "-e", choices=["vosk", "whisper"], help="Speech recognition engine to use")
    
    args = parser.parse_args()
    
    # Override engine in config if specified
    if args.engine:
        config = get_config()
        config.set("engine", args.engine)
    
    # Initialize the transcriber
    transcriber = AudioTranscriber(model_path=args.model)
    
    # Transcribe the file
    print(f"Transcribing {args.input}...")
    text = transcriber.transcribe_file(args.input)
    
    # Output the transcription
    if text:
        print("\nTranscription:")
        print(text)
        
        # Save to file if specified
        if args.output:
            with open(args.output, "w") as f:
                f.write(text)
            print(f"\nTranscription saved to {args.output}")
    else:
        print("Transcription failed.")


if __name__ == "__main__":
    main()