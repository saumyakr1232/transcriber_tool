#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Import necessary libraries
try:
    # Import config module
    from config import get_config
    # Import our transcriber classes
    from transcribers import BaseTranscriber, TranscriberFactory
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    sys.exit(1)


class AudioTranscriber:
    """Class for transcribing audio from MP4 or WAV files using offline methods.
    
    This class serves as a wrapper around the specific transcriber implementations,
    providing a consistent interface for the application.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the transcriber with a model.
        
        Args:
            model_path: Path to the model directory. If None, will use the configured model.
        """
        # Load configuration
        self.config = get_config()
        self.engine = self.config.get_engine()
        
        # Create the appropriate transcriber using the factory
        try:
            self.transcriber = TranscriberFactory.create_transcriber(self.config.config, model_path)
            print(f"Using {self.engine} engine for transcription.")
        except Exception as e:
            print(f"Error initializing transcriber here : {e}")
            sys.exit(1)
    
    def transcribe_file(self, input_path: str) -> str:
        """Transcribe an audio file (MP4 or WAV).
        
        Args:
            input_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        return self.transcriber.transcribe_file(input_path)


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