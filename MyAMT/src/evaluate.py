import argparse
import torch
from torch.utils.data import DataLoader
from model import BiLSTM
from conf.conf import Config
from utils.utils import preprocess_audio
from data.data import MusicNetDataset  

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
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(evaluate(model, test_loader, criterion))
