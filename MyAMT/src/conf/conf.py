class Config:
    def __init__(self):
        # Training parameters
        self.num_epochs = 200
        self.batch_size = 4
        self.learning_rate = 0.1

        # Feature extraction parameters
        self.sr = 44100  # Sample rate for audio files
        self.n_mfcc = 30  # Number of MFCC features to extract
        self.hop_length = int(self.sr * (1/64))  # Approximately 689

        # Model architecture parameters
        self.input_size = self.n_mfcc  # Input feature dimension (MFCC features)
        self.hidden_size = 512  # LSTM hidden layer size
        self.num_layers = 2  # Number of LSTM layers
        self.bidirectional = True  # Whether to use a bidirectional LSTM
        self.output_size = 88
        self.dropout = 0.3