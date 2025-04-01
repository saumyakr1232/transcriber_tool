#!/usr/bin/env python3

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

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

def download_and_setup_model(model_url=None, output_dir=None):
    """Download and set up a Vosk model.
    
    Args:
        model_url: URL to the model archive
        output_dir: Directory to extract the model to
    """
    # Use default URL if none provided
    if model_url is None:
        model_url = DEFAULT_MODEL_URL
    
    # Use default output directory if none provided
    if output_dir is None:
        output_dir = Path("./models")
    else:
        output_dir = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    print("\nModel setup complete!")
    print(f"Model is available in: {output_dir}")
    return True

def main():
    """Main function to run the model downloader."""
    parser = argparse.ArgumentParser(description="Download and set up a Vosk model for offline speech recognition.")
    parser.add_argument("--url", "-u", help=f"URL to the model archive (default: {DEFAULT_MODEL_URL})")
    parser.add_argument("--output", "-o", help="Directory to extract the model to (default: ./models)")
    
    args = parser.parse_args()
    
    # Download and set up the model
    download_and_setup_model(model_url=args.url, output_dir=args.output)


if __name__ == "__main__":
    main()