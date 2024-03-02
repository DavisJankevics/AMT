import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=config.bidirectional)
        self.fc = nn.Linear(config.hidden_size * 2, config.input_size)  # 2 for bidirection

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output