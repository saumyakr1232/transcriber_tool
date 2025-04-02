#!/usr/bin/env python3

import os
import sys
import json
import wave
import tempfile
from typing import Optional, Dict, Any

# Import necessary libraries
try:
    import numpy as np
    from vosk import Model, KaldiRecognizer
    import ffmpeg
    from .base_transcriber import BaseTranscriber
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    sys.exit(1)


class VoskTranscriber(BaseTranscriber):
    """Transcriber implementation using the Vosk speech recognition engine.
    
    This class handles transcription of audio files using the Vosk offline speech recognition engine.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Vosk transcriber.
        
        Args:
            config: Configuration dictionary for the transcriber
        """
        super().__init__(config)
        self.model = None
        self.model_path = None
    
    def load_model(self, model_path: str = None):
        """Load the Vosk model.
        
        Args:
            model_path: Path to the Vosk model directory. If None, will use the configured model.
        """
        # Try to find a model if not specified
        if model_path is None:
            model_path = self.config.get("models.vosk.model_path")
            
            # If still None, look in the models directory
            if model_path is None:
                models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                if os.path.exists(models_dir):
                    model_dirs = [d for d in os.listdir(models_dir) 
                                 if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("vosk-model")]
                    if model_dirs:
                        model_path = os.path.join(models_dir, model_dirs[0])
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
            self.model_path = model_path
            print("Vosk model loaded successfully.")
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
            sys.exit(1)
    
    def transcribe_wav(self, wav_path: str) -> str:
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