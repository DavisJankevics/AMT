import argparse
import torch
from torch.utils.data import DataLoader
from model.model import BiLSTM
from conf.conf import Config
from utils.utils import preprocess_audio
from data.data import MusicNetDataset  

def train(db_location):
    config = Config()
    model = BiLSTM(config)
    criterion = torch.nn.MSELoss()  # or any other suitable loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = MusicNetDataset(root_dir=db_location, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model.train()
    for epoch in range(config.num_epochs):
        for batch in train_loader:
            audio = preprocess_audio(batch['audio'])
            optimizer.zero_grad()
            output = model(audio)
            loss = criterion(output, batch['labels'])
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Bi-LSTM for Music Transcription')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    args = parser.parse_args()

    train(args.db_location)
