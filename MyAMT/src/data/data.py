import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import librosa
from utils.utils import extract_features

class MusicNetDataset(Dataset):
    def __init__(self, root_dir, split='train', sr=44100, hop_length=512):
        self.data_dir = os.path.join(root_dir, f'{split}_data')
        self.labels_dir = os.path.join(root_dir, f'{split}_labels')
        self.sr = sr
        self.hop_length = hop_length

        self.audio_files = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file = os.path.join(self.data_dir, self.audio_files[idx])
        audio, sr = librosa.load(audio_file, sr=self.sr, mono=True)
        features = extract_features(audio, sr=sr)

        label_file = os.path.join(self.labels_dir, self.audio_files[idx].replace('.wav', '.csv'))
        labels_df = pd.read_csv(label_file)

        # Calculate the total steps based on the audio duration and hop length
        total_steps = features.shape[0]

        # Initialize the label tensor with zeros
        label_tensor = torch.zeros(total_steps, dtype=torch.long) - 1  # Use -1 for "no note"

        # Process each label entry
        for _, row in labels_df.iterrows():
            start_step = int(row['start_time'] / (1000 * (self.sr / self.hop_length)))
            end_step = int(row['end_time'] / (1000 * (self.sr / self.hop_length)))
            note_value = row['note']  # Assuming 'note' is the target label you want to predict

            # Mark the presence of the note in the corresponding steps
            label_tensor[start_step:end_step] = note_value

        return {'audio': features, 'labels': label_tensor}

# Note: Adjust 'extract_features' as necessary to return MFCC features with the expected shape.
