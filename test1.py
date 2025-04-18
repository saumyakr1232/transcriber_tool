import torchaudio
import numpy as np
import torch
from sklearn.cluster import KMeans
from speechbrain.pretrained import SpeakerRecognition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def extract_embeddings(wav_path):
    """
    Extract speaker embeddings from a WAV file.

    Parameters:
        wav_path (str): Path to the audio file (WAV format).

    Returns:
        embeddings (tensor): The extracted speaker embeddings.
    """

    # Load pre-trained speaker recognition model from SpeechBrain
    speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmpdir")

    # Load audio file
    signal, sample_rate = torchaudio.load(wav_path)

    # Ensure the audio is mono (single-channel)
    if signal.ndim > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # Extract embeddings from the audio signal
    embeddings = speaker_model.encode_batch(signal)

    # Return as a numpy array
    return embeddings.cpu().numpy()


def diarize_speakers(wav_path, num_speakers=2):
    """
    Perform speaker diarization by extracting speaker embeddings and clustering.

    Parameters:
        wav_path (str): Path to the audio file (WAV format).
        num_speakers (int): Number of speakers to identify in the audio.

    Returns:
        None: Displays the diarization results.
    """

    # Step 1: Extract embeddings from the audio file
    embeddings = extract_embeddings(wav_path)

    # Step 2: Standardize the embeddings (important for clustering)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Step 3: Apply K-Means clustering to the embeddings to identify speakers
    kmeans = KMeans(n_clusters=num_speakers, random_state=42)
    speaker_labels = kmeans.fit_predict(embeddings_scaled)

    # Step 4: Print and visualize the diarization results
    print(f"Number of segments: {len(speaker_labels)}")
    print(f"Identified speakers: {num_speakers}")
    print(f"Speaker assignments for each segment: {speaker_labels}")

    # Visualization of clustering (optional, only works well if num_speakers is <= 10)
    plt.scatter(np.arange(len(speaker_labels)), np.zeros(len(speaker_labels)), c=speaker_labels, cmap='viridis')
    plt.title("Speaker Diarization")
    plt.xlabel("Segment Index")
    plt.ylabel("Speaker Label")
    plt.show()


# Example usage
if __name__ == "__main__":
    wav_file_path = "/Users/saumya/Downloads/Podcast_convo.wav"  # Replace with the path to your WAV file
    diarize_speakers(wav_file_path, num_speakers=2)  # You can adjust num_speakers as neede
