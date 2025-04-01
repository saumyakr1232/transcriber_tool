# Audio Capture Tool

This tool allows you to capture audio from both your system's output (what you hear) and your microphone simultaneously, and merge them into a single WAV file.

## Features

- Record system audio output (e.g., YouTube videos, music, etc.)
- Record microphone input simultaneously
- Merge both audio streams into a single WAV file
- Adjustable recording duration
- Configurable sample rate

## Requirements

- Python 3.6+
- sounddevice
- numpy
- pydub

## Installation

Ensure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Platform-specific requirements

#### macOS

To capture system audio on macOS, you may need to install additional software like BlackHole or Soundflower to create a virtual audio device.

#### Windows

On Windows, you may need to enable "Stereo Mix" in your sound settings to capture system audio.

#### Linux

On Linux, you may need to configure PulseAudio or ALSA to allow system audio capture.

## Usage

### Command Line

```bash
python audio_capture.py --output captured_audio.wav --duration 30
```

Options:

- `--output`, `-o`: Path to the output WAV file (default: captured_audio.wav)
- `--duration`, `-d`: Duration to record in seconds (default: 10.0)
- `--sample-rate`, `-r`: Sample rate in Hz (default: 44100)
- `--list-devices`, `-l`: List available audio devices and exit

### As a Module

```python
from audio_capture import AudioCapture

# Create an audio capture instance
audio_capture = AudioCapture()

# Start recording
audio_capture.start_recording()

# ... do something while recording ...

# Stop recording
system_audio, mic_audio = audio_capture.stop_recording()

# Save merged audio
output_path = audio_capture.save_merged_audio("output.wav")
```

## Troubleshooting

### No system audio is being captured

- Make sure you have the correct audio output device selected
- On macOS, ensure you have a virtual audio device set up
- On Windows, ensure "Stereo Mix" is enabled

### Poor audio quality

- Try increasing the sample rate using the `--sample-rate` option
- Adjust the mixing weights in the `merge_audio` method if one source is too loud or quiet

### List available devices

To see all available audio devices on your system:

```bash
python audio_capture.py --list-devices
```
