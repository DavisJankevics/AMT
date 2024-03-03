import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,  # Number of MFCC features
            hidden_size=config.hidden_size,  # LSTM hidden layer size
            num_layers=config.num_layers,  # Number of LSTM layers
            bidirectional=config.bidirectional,  # Use bidirectional LSTM
            batch_first=True  # Input and output tensors are provided as (batch, seq, feature)
        )
        # The output layer that maps from hidden state space to class space
        self.fc = nn.Linear(
            config.hidden_size * 2 if config.bidirectional else config.hidden_size,
            config.output_size  # Number of output classes (e.g., MIDI note numbers)
        )

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_x)
        
        # Unpack the sequence
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply the fully connected layer to each time step
        output = self.fc(output)
        
        return output