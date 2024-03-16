import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from model.model import BiLSTM
from conf.conf import Config
from utils.utils import preprocess_audio
from data.data import MusicNetDataset  
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, accuracy_score, f1_score

def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint file in the specified directory.

    Args:
    - checkpoint_dir: Path to the directory containing checkpoint files.

    Returns:
    - The path to the latest checkpoint file.
    """
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(load_path, model, device):
    """
    Loads model weights from a checkpoint file.

    Args:
    - load_path: Path to the checkpoint file.
    - model: The model to load weights into.
    - device: The device to perform computation on ('cuda' or 'cpu').
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from <== {load_path}')


def custom_collate_fn(batch):
    # Extract features and labels from the batch
    audio_features = [item['audio'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad the sequences of features
    audio_features_padded = pad_sequence(audio_features, batch_first=True).to(device)

    # Since labels could be of different lengths, also pad them
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1).to(device)  # Assuming -1 is an ignore index

    # Calculate lengths of sequences before padding for dynamic unpadding later if necessary
    lengths = torch.tensor([len(feature) for feature in audio_features], dtype=torch.long).to(device)

    return {'audio': audio_features_padded, 'labels': labels_padded, 'lengths': lengths}

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            audio_features = batch['audio'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)

            # Forward pass
            output = model(audio_features, lengths)
            output = output.transpose(1, 2)  # Assuming [batch, num_classes, seq_len]

            # Loss
            loss = criterion(output, labels)
            total_loss += loss.item()

            # Predictions
            _, predicted = torch.max(output, dim=1)
            all_predictions.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    # Flatten all predictions and labels to compute overall metrics
    all_predictions = np.concatenate(all_predictions).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Remove padding index (-1) from labels and predictions
    valid_indices = all_labels != -1  # Assuming -1 is your padding value
    valid_labels = all_labels[valid_indices]
    valid_predictions = all_predictions[valid_indices]

    # Calculate metrics
    precision = precision_score(valid_labels, valid_predictions, average='weighted')
    accuracy = accuracy_score(valid_labels, valid_predictions)
    f1 = f1_score(valid_labels, valid_predictions, average='weighted')

    return total_loss / len(loader), precision, accuracy, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Bi-LSTM for Music Transcription')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory containing model checkpoints')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    config = Config()
    model = BiLSTM(config)
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    load_checkpoint(latest_checkpoint, model, device)
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Assuming -1 is used for padding
    test_dataset = MusicNetDataset(root_dir=args.db_location, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    loss, precision, accuracy, f1 = evaluate(model, test_loader, criterion, device)
    print(f'Loss: {loss}, Precision: {precision}, Accuracy: {accuracy}, F1 Score: {f1}')
