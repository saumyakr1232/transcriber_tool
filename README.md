# Audio Transcription Tool

A Python application that transcribes text from MP4 or WAV files using offline methods.

## Features

- Transcribe audio from MP4 and WAV files
- Works completely offline
- Simple command-line interface
- Outputs transcription to text files

## Requirements

- Python 3.8+
- FFmpeg (for MP4 processing)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Vosk model (for offline speech recognition):
   - Visit https://alphacephei.com/vosk/models
   - Download a model appropriate for your language (e.g., vosk-model-small-en-us-0.15)
   - Extract the model to the `models` directory

## Usage

```bash
python transcriber.py --input your_audio_file.mp4 --output transcription.txt
```

Or use the simplified interface:

```bash
python app.py
```

## License

MIT
