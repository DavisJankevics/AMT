import argparse
import torch
from torch.utils.data import DataLoader
from model.model import BiLSTM
from conf.conf import Config
from data.data import MusicNetDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

def save_checkpoint(save_path, model, optimizer, epoch, loss):
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state_dict, save_path)
    print(f'Model saved to => {save_path}')

def load_checkpoint(load_path, model, optimizer, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Model loaded from <== {load_path}')
    return model, optimizer, epoch, loss

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

def train(db_location, load_model_path=None):
    config = Config()
    model = BiLSTM(config).to(device)

    start_epoch = 0
    if load_model_path is not None:
        model, optimizer, start_epoch, _ = load_checkpoint(load_model_path, model, optimizer, device)

    
    # Adjust loss function for classification tasks, assuming ignore_index=-1 for padded values
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = MusicNetDataset(root_dir=db_location, split='train', sr=config.sr, hop_length=config.hop_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    model.train()
    for epoch in range(config.num_epochs):
        print(f'Starting Epoch {epoch+1}')
        for batch_idx, batch in enumerate(train_loader):
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}')

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

            if epoch % 10 == 0:  # Assuming you want to save every 10 epochs
                save_path = f'checkpoint_epoch_{epoch}.pth'
                save_checkpoint(save_path, model, optimizer, epoch, loss.item())

            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BiLSTM for Music Transcription')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load the model checkpoint')
    args = parser.parse_args()

    train(args.db_location, args.load_model_path)
