# import torch
# from torch.utils.data import Dataset
import os
import pandas as pd
import librosa
# from utils.utils import extract_features

# class MusicNetDataset(Dataset):
#     def __init__(self, root_dir, split='train', sr=44100, hop_length=512, n_mfcc=13):
#         self.data_dir = os.path.join(root_dir, f'{split}_data')
#         self.labels_dir = os.path.join(root_dir, f'{split}_labels')
#         self.sr = sr
#         self.hop_length = hop_length
#         self.n_mfcc = n_mfcc

#         # Piano range
#         self.min_midi = 21
#         self.max_midi = 108
#         self.num_notes = 88  # Fixed to piano range

#         self.audio_files = sorted(os.listdir(self.data_dir))

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         audio_file = os.path.join(self.data_dir, self.audio_files[idx])
#         audio, sr = librosa.load(audio_file, sr=self.sr, mono=True)
#         features = extract_features(audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)

#         label_file = os.path.join(self.labels_dir, self.audio_files[idx].replace('.wav', '.csv'))
#         labels_df = pd.read_csv(label_file)

#         total_steps = features.shape[0]
        
#         # Initialize the label tensor with -1 for each time step
#         label_tensor = torch.zeros((total_steps, self.num_notes), dtype=torch.float32) - 1

#         for _, row in labels_df.iterrows():
#             start_time = row['start_time'] / 1000.0  # Convert to seconds
#             end_time = row['end_time'] / 1000.0
#             start_step = int(start_time * self.sr / self.hop_length)
#             end_step = int(end_time * self.sr / self.hop_length)
#             note = int(row['note'])
#             note_index = note - self.min_midi  # Convert MIDI number to index
            
#             # Set labels for active note in its duration
#             if 0 <= note_index < self.num_notes:
#                 label_tensor[start_step:end_step, note_index] = 1

#         return {'audio': features, 'labels': label_tensor}
import tensorflow as tf
import pandas as pd
import os
import librosa
import numpy as np

# def load_data_and_labels(audio_file_path, label_file_path, sr=44100, hop_length=512, n_mfcc=13, target_duration=7.5):
#     audio, sr = librosa.load(audio_file_path, sr=sr, mono=True)
    
#     # Calculate the target length in frames
#     target_length = int(sr * target_duration / hop_length)
    
#     mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T
#     mfcc_features = np.expand_dims(mfcc_features, -1)  # Add channel dimension
    
#     # Pad or truncate the mfcc features to match the target_length
#     if len(mfcc_features) < target_length:
#         # Pad
#         padding = np.zeros((target_length - len(mfcc_features), n_mfcc, 1))
#         mfcc_features = np.concatenate((mfcc_features, padding), axis=0)
#     else:
#         # Truncate
#         mfcc_features = mfcc_features[:target_length]
def load_audio_and_labels(audio_file_path, label_file_path, sr=44100, hop_length=512, n_fft=2048, n_mels=229, target_duration=10):
    target_length = int(sr * target_duration / hop_length)  # Target length in frames

    # Load and process audio file to Mel spectrogram
    audio, _ = librosa.load(audio_file_path, sr=sr, mono=True, duration=target_duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    max_start_index = max(mel_spec_norm.shape[1] - target_length, 0)  # Ensure non-negative
    start_index = np.random.randint(0, max_start_index + 1) if max_start_index > 0 else 0
    end_index = start_index + target_length
    # Ensure the Mel spectrogram has the target shape
    if mel_spec_norm.shape[1] < target_length:
        padding = np.zeros((n_mels, target_length - mel_spec_norm.shape[1]))
        mel_spec_norm = np.concatenate((mel_spec_norm, padding), axis=1)
    else:
        mel_spec_norm = mel_spec_norm[:, start_index:end_index]
    labels_df = pd.read_csv(label_file_path)
    label_tensor = np.zeros((target_length, 88), dtype=np.float32)  # Initialize label tensor with zeros
    start_time_offset = start_index * hop_length / sr  # Start time in seconds of the cropped spectrogram

    for _, row in labels_df.iterrows():
        original_start_time = row['start_time'] / sr
        original_end_time = row['end_time'] / sr

        # Adjust times based on the start_time_offset
        adjusted_start_time = original_start_time - start_time_offset
        adjusted_end_time = original_end_time - start_time_offset

        # Convert times to steps
        start_step = max(int(adjusted_start_time * sr / hop_length), 0)
        end_step = min(int(adjusted_end_time * sr / hop_length), target_length)

        note = int(row['note']) - 21

        # Mark note as active for its duration
        if 0 <= note < 88 and start_step < end_step:
            label_tensor[start_step:end_step, note] = 1
    # print("label_tensor",label_tensor.shape)
    # print("mel_spec_norm",mel_spec_norm.T.shape)
    return mel_spec_norm.T, label_tensor

def create_tf_dataset(root_dir, split, sr=44100, hop_length=512, n_fft=2048, n_mels=229, target_duration=10):
    data_dir = os.path.join(root_dir, f'{split}_data')
    labels_dir = os.path.join(root_dir, f'{split}_labels')
    audio_files = sorted(os.listdir(data_dir))
    
    def gen():
        for audio_file in audio_files:
            audio_path = os.path.join(data_dir, audio_file)
            label_path = os.path.join(labels_dir, audio_file.replace('.wav', '.csv'))
            features, labels = load_audio_and_labels(audio_path, label_path, sr, hop_length, n_fft, n_mels, target_duration)
            yield (features, labels)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, n_mels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 88), dtype=tf.float32),
        ))
