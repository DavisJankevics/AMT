class Config:
    def __init__(self):
        # Training parameters
        self.num_epochs = 100
        self.batch_size = 64
        self.learning_rate = 0.001

        # Feature extraction parameters
        self.sr = 44100  # Sample rate for audio files
        self.n_mfcc = 20  # Number of MFCC features to extract
        # Assuming a hop length that matches the quantization of 1/64 of a second at 44100 Hz sample rate
        self.hop_length = int(self.sr * (1/64))  # Approximately 689

        # Model architecture parameters
        self.input_size = self.n_mfcc  # Input feature dimension (MFCC features)
        self.hidden_size = 256  # LSTM hidden layer size
        self.num_layers = 2  # Number of LSTM layers
        self.bidirectional = True  # Whether to use a bidirectional LSTM
        self.output_size = 103  # Number of output classes