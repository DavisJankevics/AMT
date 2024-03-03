import argparse
import torch
from torch.utils.data import DataLoader
from model import BiLSTM
from conf.conf import Config
from utils.utils import preprocess_audio
from data.data import MusicNetDataset  
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    # Separate audio and labels
    audio_batch = [item['audio'] for item in batch]
    labels_batch = [item['labels'] for item in batch]
    
    # Pad audio sequences to be the same length (if not already handled in __getitem__)
    # audio_batch_padded = pad_sequence(audio_batch, batch_first=True, padding_value=0)
    
    # Pad label sequences to the same length
    labels_batch_padded = pad_sequence(labels_batch, batch_first=True, padding_value=-1)  # Use an appropriate padding value for labels
    
    return {'audio': torch.stack(audio_batch, dim=0), 'labels': labels_batch_padded}

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            audio = preprocess_audio(batch['audio'])
            output = model(audio)
            loss = criterion(output, batch['labels'])
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Bi-LSTM for Music Transcription')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    args = parser.parse_args()

    config = Config()
    model = BiLSTM(config)
    criterion = torch.nn.MSELoss()  # Define the loss function
    test_dataset = MusicNetDataset(root_dir=args.db_location, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    print(evaluate(model, test_loader, criterion))
