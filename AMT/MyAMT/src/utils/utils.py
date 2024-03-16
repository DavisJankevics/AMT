import librosa
import torch
import numpy as np
from midiutil import MIDIFile
from conf.conf import Config

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


# def postprocess_output(output_tensor, device, output_path="output.mid"):
#     # Assuming `output` is a sequence of note values or an equivalent representation
#     print(f'Writing output to {output_tensor}')
#     print(f'Writing output to {output_path}')
#     # Convert logits to predicted note values (assuming argmax for simplicity)
#     _, predicted_notes = torch.max(output_tensor, dim=2)  # Assuming [batch, time, notes]

#     midi = MIDIFile(1)  # One track
#     track = 0
#     time = 0  # Start at the beginning
#     channel = 0
#     volume = 100

#     midi.addTrackName(track, time, "Sample Track")
#     midi.addTempo(track, time, 120)  # Example tempo

#     # Placeholder for actual note duration and start time calculation
#     duration = 1  # Fixed duration for simplicity

#     # Iterate over predicted notes to add to MIDI file
#     # Here we simplify by assigning each note a fixed duration and stepping through time
#     for note_time, note in enumerate(predicted_notes[0]):  # Assuming batch size of 1
#         if note.item() > 0:  # Assuming 0 is a 'no note' class
#             pitch = note.item()  # This is a simplification
#             midi.addNote(track, channel, pitch, note_time, duration, volume)

#     with open(output_path, "wb") as f:
#         midi.writeFile(f)
#     print(f'MIDI file written to {output_path}')

def postprocess_output(output_tensor, device, output_path="output.mid"):
    """
    Converts the model's output tensor to a MIDI file, taking into account the provided configuration.

    Args:
    - output_tensor: The output tensor from the model.
    - device: The computation device.
    - output_path: Path to save the generated MIDI file.
    - config: Configuration object containing parameters like sample rate and hop length.
    """

    print(f'Writing output to {output_path}')

    # Convert logits to predicted note values (assuming argmax for simplicity)
    a, predicted_notes = torch.max(output_tensor, dim=2)  # Assuming [batch, time, notes]
    print(f'\na {a}\n')
    # print(f'\nb {b}\n')
    print(f'\npredicted_notes {predicted_notes}\n')
    
    config = Config()
    midi = MIDIFile(1)  # One track
    track = 0
    time = 0  # Start at the beginning
    channel = 0
    volume = 100

    midi.addTrackName(track, time, "Sample Track")
    midi.addTempo(track, time, 120)  # Using a fixed tempo for simplicity

    # Calculate the time in seconds for each frame based on hop length and sample rate
    frame_duration = config.hop_length / float(config.sr)  # Duration of each frame in seconds

    for i, note in enumerate(predicted_notes[0]):  # Loop through each time step
        if note.item() > 0:  # Assuming 0 is a 'no note' or 'rest' class
            pitch = note.item()
            start_time = i * frame_duration  # Calculate the start time for this note
            duration = frame_duration  # Assigning a fixed duration equal to the frame duration

            # Only add MIDI note if pitch is within MIDI range
            if 0 < pitch <= 127:
                midi.addNote(track, channel, pitch, start_time, duration, volume)

    with open(output_path, "wb") as f:
        midi.writeFile(f)
    print(f'MIDI file written to {output_path}')