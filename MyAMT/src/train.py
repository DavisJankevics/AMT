import argparse
import torch
from torch.utils.data import DataLoader
from model.model import BiLSTM
from conf.conf import Config
from data.data import MusicNetDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def train(db_location):
    config = Config()
    model = BiLSTM(config).to(device)
    
    # Adjust loss function for classification tasks, assuming ignore_index=-1 for padded values
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = MusicNetDataset(root_dir=db_location, split='train', sr=config.sr, hop_length=config.hop_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    model.train()
    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            audio_features = batch['audio']
            labels = batch['labels']
            lengths = batch['lengths']
            # max_note_value = max(labels)  # Assuming `labels` is a list/array of your label values
            # if max_note_value >= config.output_size:
            #     raise ValueError(f"Found label value {max_note_value} exceeding configured output size {config.output_size}.")

            optimizer.zero_grad()

            # Assuming model's forward method is designed to handle packed sequences
            output = model(audio_features, lengths)
            output = output.transpose(1, 2)  # Adjusting dimensions for CrossEntropyLoss
            
            # CrossEntropyLoss expects labels of shape [batch_size, seq_len] without padding
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BiLSTM for Music Transcription')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    args = parser.parse_args()

    train(args.db_location)
