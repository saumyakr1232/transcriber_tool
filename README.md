# Audio Transcription Tool

## Requirements

- Python 3.8+
- FFmpeg (for MP4 processing)
- Required Python packages (see requirements.txt)
- CUDA-compatible GPU (optional, for faster Whisper transcription)
- Ollama (required for text summarization feature)

## Installing Ollama

Ollama is required for the text summarization feature. Follow these steps to install it:

### macOS

```bash
brewcurl -fsSL https://ollama.com/install.sh | sh
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download and install from: https://ollama.com/download/windows

or

```bash
winget install ollama
```

### Verify Installation

After installation, verify Ollama is working:

```bash
ollama run mistral "Hello"
```

This will download the mistral model (if not already present) and verify it's working correctly.

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up a speech recognition model:

   ### Option 1: Vosk (Default)

   - Visit https://alphacephei.com/vosk/models
   - Download a model appropriate for your language (e.g., vosk-model-small-en-us-0.15)
   - Extract the model to the `models` directory
   - Or use the helper script: `python download_model.py --engine vosk`

   ### Option 2: Whisper

   - No manual download required - models are downloaded automatically on first use
   - Configure which model to use in the application settings
   - Or use the helper script: `python download_model.py --engine whisper --model-size base`

## Usage

### GUI Application

Run the application with:

```bash
python app.py
# or
./run.sh
```

You can specify which engine to use when running the application:

```bash
# Run with Vosk engine
./run.sh --engine vosk

# Run with Whisper engine
./run.sh --engine whisper
```

### Command Line Usage

You can also use the transcriber directly from the command line:

```bash
# Using Vosk engine
python transcriber.py --input your_audio_file.mp4 --output transcription.txt --engine vosk

# Using Whisper engine
python transcriber.py --input your_audio_file.mp4 --output transcription.txt --engine whisper
```

## Configuration

The application uses a configuration file (`config.json`) to store settings. You can edit this file directly or use the Settings dialog in the application.

### Available Configuration Options

```json
{
  "engine": "vosk", // Speech recognition engine: "vosk" or "whisper"
  "models": {
    "vosk": {
      "model_path": "./models/vosk-model-small-en-us-0.15" // Path to Vosk model
    },
    "whisper": {
      "model_size": "base", // Whisper model size: tiny, base, small, medium, large
      "use_gpu": true, // Use GPU for Whisper if available
      "language": "" // Language code (empty for auto-detection)
    }
  },
  "transcription": {
    "sample_rate": 16000, // Audio sample rate
    "word_timestamps": false // Show word timestamps (Vosk only)
  }
}
```

## Engines Comparison

### Vosk

- Pros: Faster, lower resource usage, works entirely offline
- Cons: Less accurate for some languages, requires downloading models manually

### Whisper

- Pros: More accurate, supports many languages, better handling of accents and background noise
- Cons: Slower (especially without GPU), higher resource usage, downloads models on first use
