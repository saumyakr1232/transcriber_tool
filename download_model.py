import os
import sys
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Import our config module
try:
    from config import get_config
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please make sure config.py is in the same directory as this script.")
    sys.exit(1)

# Try to import whisper if available
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def download_vosk_model(model_name=None):
    """Download a Vosk model.
    
    Args:
        model_name: Name of the model to download. If None, will download the default model.
    """
    # Default model if not specified
    if model_name is None:
        model_name = "vosk-model-small-en-us-0.15"
    
    # Model URL
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    
    # Create models directory if it doesn't exist
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Download the model
    print(f"Downloading Vosk model: {model_name}")
    print(f"URL: {model_url}")
    print("This may take a while...")
    
    # Download to a temporary file
    temp_zip = models_dir / f"{model_name}.zip"
    try:
        urllib.request.urlretrieve(model_url, temp_zip)
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False
    
    # Extract the model
    print(f"Extracting model to {models_dir}")
    try:
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Remove the zip file
        os.remove(temp_zip)
        
        # Update the config
        config = get_config()
        config.set("models.vosk.model_path", str(models_dir / model_name))
        config.save()
        
        print(f"Model downloaded and extracted successfully to {models_dir / model_name}")
        print("Configuration updated.")
        return True
    except Exception as e:
        print(f"Error extracting model: {e}")
        return False


def download_whisper_model(model_size="base"):
    """Download a Whisper model.
    
    Args:
        model_size: Size of the model to download (tiny, base, small, medium, large).
    """
    if not WHISPER_AVAILABLE:
        print("Error: Whisper is not available.")
        print("Please install Whisper with: pip install openai-whisper")
        return False
    
    # Check if model size is valid
    valid_sizes = ["tiny", "base", "small", "medium", "large"]
    if model_size not in valid_sizes:
        print(f"Error: Invalid model size '{model_size}'")
        print(f"Valid sizes: {', '.join(valid_sizes)}")
        return False
    
    # Download the model
    print(f"Downloading Whisper model: {model_size}")
    print("This may take a while...")
    
    try:
        # This will download the model to the whisper cache directory
        whisper.load_model(model_size)
        
        # Update the config
        config = get_config()
        config.set("models.whisper.model_size", model_size)
        config.save()
        
        print(f"Model '{model_size}' downloaded successfully")
        print("Configuration updated.")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Download speech recognition models.")
    parser.add_argument("--engine", "-e", choices=["vosk", "whisper"], default="vosk",
                        help="Speech recognition engine to download model for")
    parser.add_argument("--model-name", "-n", help="Name of the Vosk model to download")
    parser.add_argument("--model-size", "-s", choices=["tiny", "base", "small", "medium", "large"],
                        default="base", help="Size of the Whisper model to download")
    
    args = parser.parse_args()
    
    if args.engine == "vosk":
        download_vosk_model(args.model_name)
    elif args.engine == "whisper":
        if not WHISPER_AVAILABLE:
            print("Error: Whisper is not available.")
            print("Please install Whisper with: pip install openai-whisper")
            sys.exit(1)
        download_whisper_model(args.model_size)


if __name__ == "__main__":
    main()