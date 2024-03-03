import librosa
import torch
import numpy as np
from midiutil import MIDIFile

def extract_features(audio, sr=44100, n_mfcc=20, hop_length=512):
    """
    Extract MFCC features from a single audio file.
    
    Parameters:
    - audio: np.array, raw audio data.
    - sr: int, sample rate.
    - n_mfcc: int, number of MFCC features to extract.
    - hop_length: int, the number of samples between successive frames, e.g., for time quantization.
    
    Returns:
    - mfcc_features: torch.Tensor, MFCC features with shape (seq_len, n_mfcc).
    """
    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # Transpose to shape (seq_len, n_mfcc)
    mfcc_features = np.transpose(mfcc_features)
    # Convert to torch tensor and ensure it's a float
    mfcc_features = torch.tensor(mfcc_features).float()
    return mfcc_features

def preprocess_audio(audio_path, sr=44100, n_mfcc=20, hop_length=512):
    """
    Load an audio file, extract MFCC features, and return as a torch tensor.
    
    Parameters:
    - audio_path: str, path to the audio file.
    - sr: int, sample rate to load the audio.
    - n_mfcc: int, number of MFCC features to extract.
    - hop_length: int, the number of samples between successive frames, e.g., for time quantization.
    
    Returns:
    - mfcc_features: torch.Tensor, MFCC features with shape (seq_len, n_mfcc).
    """
    # Load the audio file
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    # Extract MFCC features from the audio
    mfcc_features = extract_features(audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc_features


def postprocess_output(output, output_path="output.mid"):
    # Assuming `output` is a sequence of note values or an equivalent representation
    midi = MIDIFile(1)  # One track
    track = 0
    time = 0  # Start at the beginning
    channel = 0
    volume = 100

    midi.addTrackName(track, time, "Track")
    midi.addTempo(track, time, 120)  # Set a tempo

    # Example of adding notes: midi.addNote(track, channel, pitch, time, duration, volume)
    # Here you'd iterate over your `output` to add notes accordingly
    for note_info in output:
        pitch = note_info['pitch']  # MIDI note number
        start_time = note_info['start_time']  # Start time in beats
        duration = note_info['duration']  # Duration in beats
        midi.addNote(track, channel, pitch, start_time, duration, volume)

    with open(output_path, "wb") as f:
        midi.writeFile(f)