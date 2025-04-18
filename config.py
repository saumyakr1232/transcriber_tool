import os
import json
from pathlib import Path
from typing import Any

# Default configuration
DEFAULT_CONFIG = {
    # Speech recognition engine: 'vosk' or 'whisper'
    "engine": "vosk",

    # Model paths
    "models": {
        # Vosk model settings
        "vosk": {
            # Path to Vosk model directory (relative to project root or absolute)
            "model_path": "./models/vosk-model-small-en-us-0.15",
            # Alternative models can be added here
        },

        # Whisper model settings
        "whisper": {
            # Whisper model size: tiny, base, small, medium, large
            "model_size": "base",
            # Use GPU for Whisper if available
            "use_gpu": True,
            # Language code (set to 'en' for English)
            "language": "en",
        }
    },

    # Transcription settings
    "transcription": {
        # Audio settings
        "sample_rate": 16000,
        # Show word timestamps
        "word_timestamps": False,
    },

    # Speaker diarization settings
    "diarization": {
        # Enable speaker diarization
        "enabled": True,
        # HuggingFace token for accessing pyannote.audio models
        "hf_token": "",
        # Path to local diarization model (if available)
        "model_path": ""
    }
}


class Config:
    """Configuration manager for the transcriber application."""

    def __init__(self, config_path=None):
        """Initialize the configuration.

        Args:
            config_path: Path to the configuration file. If None, will look for config.json
                         in the project root directory.
        """
        self.config_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Set default config path if not provided
        if config_path is None:
            self.config_path = self.config_dir / "config.json"
        else:
            self.config_path = Path(config_path)

        # Load configuration
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from file or create default if not exists."""
        # Check if config file exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                print(f"Configuration loaded from {self.config_path}")
                return config
            except Exception as e:
                print(f"Error loading configuration: {e}")
                print("Using default configuration")
                return DEFAULT_CONFIG.copy()
        else:
            # Create default configuration file
            self._save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

    def _save_config(self, config):
        """Save configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def save(self):
        """Save current configuration to file."""
        self._save_config(self.config)

    def get(self, key, default=None) -> Any:
        """Get a configuration value.

        Args:
            key: The configuration key (can use dot notation for nested keys)
            default: Default value if key not found

        Returns:
            The configuration value or default if not found
        """
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if part in value:
                    value = value[part]
                else:
                    return default
            return value
        else:
            return self.config.get(key, default)

    def set(self, key, value):
        """Set a configuration value.

        Args:
            key: The configuration key (can use dot notation for nested keys)
            value: The value to set
        """
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.config[key] = value

    def get_engine(self) -> str:
        """Get the configured speech recognition engine.

        Returns:
            The engine name ('vosk' or 'whisper')
        """
        return self.get("engine", "vosk")

    def get_model_path(self):
        """Get the model path for the current engine.

        Returns:
            The model path for the current engine
        """
        engine = self.get_engine()
        if engine == "vosk":
            return self.get(f"models.{engine}.model_path")
        elif engine == "whisper":
            return self.get(f"models.{engine}.model_size")
        return None


# Create a global configuration instance
config: Config = Config()


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        The global Config instance
    """
    return config


if __name__ == "__main__":
    # If run directly, print the current configuration
    import pprint
    cfg: Config = get_config()
    print("Current configuration:")
    pprint.pprint(cfg.config)
    print(cfg.get("models.whisper.language"))
