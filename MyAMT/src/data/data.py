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

        audio_file = os.path.join(self.data_dir, self.audio_files[idx])
        audio, sr = librosa.load(audio_file, sr=None, mono=False)  # load audio as a numpy array

        # Pad or truncate the audio data to a fixed length
        max_length = 44100 * 30  # e.g., 30 seconds at 44100 Hz
        if audio.ndim == 1:  # mono audio
            if len(audio) < max_length:
                padding = np.zeros(max_length - len(audio))
                audio = np.concatenate((audio, padding))
            else:
                audio = audio[:max_length]
        else:  # stereo audio
            if audio.shape[1] < max_length:
                padding = np.zeros((2, max_length - audio.shape[1]))
                audio = np.concatenate((audio, padding), axis=1)
            else:
                audio = audio[:, :max_length]

        audio = torch.from_numpy(audio)  # convert to PyTorch tensor

        # Load the corresponding labels file
        label_file = os.path.join(self.labels_dir, self.audio_files[idx].replace('.wav', '.csv'))
        labels = pd.read_csv(label_file)  # load labels as a pandas DataFrame

        # Convert labels to a suitable format here, if necessary

        if self.transform:
            audio = self.transform(audio)

        return {'audio': audio, 'labels': labels}
