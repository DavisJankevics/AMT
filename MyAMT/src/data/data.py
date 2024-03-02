import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import librosa

class MusicNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.data_dir = os.path.join(root_dir, f'{split}_data')
        self.labels_dir = os.path.join(root_dir, f'{split}_labels')
        self.transform = transform

        # Get a sorted list of audio files
        self.audio_files = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the audio file
        audio_file = os.path.join(self.data_dir, self.audio_files[idx])
        audio, sr = librosa.load(audio_file, sr=None, mono=False)  # load audio as a numpy array
        audio = torch.from_numpy(audio)  # convert to PyTorch tensor

        # Load the corresponding labels file
        label_file = os.path.join(self.labels_dir, self.audio_files[idx].replace('.wav', '.csv'))
        labels = pd.read_csv(label_file)  # load labels as a pandas DataFrame

        # Convert labels to a suitable format here, if necessary

        if self.transform:
            audio = self.transform(audio)

        return {'audio': audio, 'labels': labels}
