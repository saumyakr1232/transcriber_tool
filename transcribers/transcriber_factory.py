#!/usr/bin/env python3

import os
import sys
from typing import Dict, Any, Optional

# Import our transcriber classes
try:
    from .base_transcriber import BaseTranscriber
    from .vosk_transcriber import VoskTranscriber
    from .whisper_transcriber import WhisperTranscriber
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install the required dependencies using: pip install -r requirements.txt")
    sys.exit(1)


class TranscriberFactory:
    """Factory class for creating transcriber instances.
    
    This class is responsible for creating the appropriate transcriber instance
    based on the configuration. It supports creating VoskTranscriber and WhisperTranscriber
    instances.
    """
    
    @staticmethod
    def create_transcriber(config: Dict[str, Any], model_path: Optional[str] = None) -> BaseTranscriber:
        """Create a transcriber instance based on the configuration.
        
        Args:
            config: Configuration dictionary
            model_path: Optional path to the model. If provided, overrides the config.
            
        Returns:
            A transcriber instance (VoskTranscriber or WhisperTranscriber)
            
        Raises:
            ValueError: If the engine specified in the config is not supported
        """
        # Get the engine from the config
        engine = config.get("engine", "vosk")
        
        # Create the appropriate transcriber
        if engine == "vosk":
            return VoskTranscriber(config)
        elif engine == "whisper":
            return WhisperTranscriber(config)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    
    @staticmethod
    def get_available_engines() -> list:
        """Get a list of available transcription engines.
        
        Returns:
            A list of available engine names
        """
        engines = ["vosk"]
        
        # Check if Whisper is available
        try:
            import whisper
            engines.append("whisper")
        except ImportError:
            pass
        
        return engines