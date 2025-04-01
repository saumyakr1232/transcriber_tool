#!/usr/bin/env python3

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

# Import config module
try:
    from config import get_config
except ImportError:
    print("Error: Could not import config module.")
    print("Make sure config.py is in the same directory as this script.")
    sys.exit(1)

# Default model URL (small English model)
DEFAULT_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

def download_file(url, output_path):
    """Download a file from a URL to the specified path.
    
    Args:
        url: The URL to download from
        output_path: The path to save the file to
    """
    print(f"Downloading {url} to {output_path}...")
    try:
        # Create a progress bar
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            sys.stdout.write(f"\rProgress: {percent}% [{downloaded} / {total_size} bytes]")
            sys.stdout.flush()
        
        # Download the file
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def extract_archive(archive_path, output_dir):
    """Extract an archive file to the specified directory.
    
    Args:
        archive_path: Path to the archive file
        output_dir: Directory to extract to
    """
    print(f"Extracting {archive_path} to {output_dir}...")
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract based on file extension
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"Unsupported archive format: {archive_path}")
            return False
        
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False

def download_and_setup_model(model_url=None, output_dir=None, engine="vosk", model_size=None):
    """Download and set up a speech recognition model.
    
    Args:
        model_url: URL to the model archive (for Vosk)
        output_dir: Directory to extract the model to
        engine: Speech recognition engine ('vosk' or 'whisper')
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    """
    # Use default output directory if none provided
    if output_dir is None:
        output_dir = Path("./models")
    else:
        output_dir = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different engines
    if engine.lower() == "vosk":
        # Use default URL if none provided
        if model_url is None:
            model_url = DEFAULT_MODEL_URL
        
        # Get the filename from the URL
        filename = os.path.basename(model_url)
        download_path = output_dir / filename
        
        # Download the model
        if not download_file(model_url, download_path):
            return False
        
        # Extract the model
        if not extract_archive(download_path, output_dir):
            return False
        
        # Clean up the downloaded archive
        os.remove(download_path)
        print(f"Removed temporary file: {download_path}")
        
        print("\nVosk model setup complete!")
        print(f"Model is available in: {output_dir}")
        
        # Update config
        config = get_config()
        # Try to find the extracted model directory
        model_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("vosk-model")]
        if model_dirs:
            config.set("models.vosk.model_path", str(model_dirs[-1]))
            config.save()
            print(f"Configuration updated to use model: {model_dirs[-1]}")
        
        return True
    
    elif engine.lower() == "whisper":
        # Check if whisper is installed
        try:
            import whisper
        except ImportError:
            print("Error: Whisper not installed.")
            print("Please install Whisper with: pip install openai-whisper")
            return False
        
        # Use default model size if none provided
        if model_size is None:
            model_size = "base"
        
        # Check if model size is valid
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if model_size not in valid_sizes:
            print(f"Error: Invalid Whisper model size '{model_size}'")
            print(f"Valid sizes: {', '.join(valid_sizes)}")
            return False
        
        print(f"\nWhisper will download the {model_size} model automatically on first use.")
        print("No manual download required.")
        
        # Update config
        config = get_config()
        config.set("models.whisper.model_size", model_size)
        config.save()
        print(f"Configuration updated to use Whisper {model_size} model")
        
        return True
    
    else:
        print(f"Error: Unknown engine '{engine}'")
        print("Supported engines: 'vosk', 'whisper'")
        return False

def main():
    """Main function to run the model downloader."""
    parser = argparse.ArgumentParser(description="Download and set up a speech recognition model.")
    parser.add_argument("--url", "-u", help=f"URL to the Vosk model archive (default: {DEFAULT_MODEL_URL})")
    parser.add_argument("--output", "-o", help="Directory to extract the model to (default: ./models)")
    parser.add_argument("--engine", "-e", choices=["vosk", "whisper"], default="vosk",
                        help="Speech recognition engine to use (default: vosk)")
    parser.add_argument("--model-size", "-s", choices=["tiny", "base", "small", "medium", "large"], default="base",
                        help="Whisper model size (default: base)")
    
    args = parser.parse_args()
    
    # Download and set up the model
    download_and_setup_model(
        model_url=args.url,
        output_dir=args.output,
        engine=args.engine,
        model_size=args.model_size
    )


if __name__ == "__main__":
    main()