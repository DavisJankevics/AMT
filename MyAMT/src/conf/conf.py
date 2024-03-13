class Config:
    def __init__(self):
        # Training parameters
        self.num_epochs = 1000
        self.batch_size = 4
        self.learning_rate = 0.0001

        # Audio processing parameters
        self.sr = 44100  # Sample rate for audio files
        self.hop_length = int(self.sr * (1/64))  # Approximately 689
        self.n_fft = 2048  # FFT window size for Mel-spectrogram
        self.n_mels = 229  # Number of Mel bins
        self.target_duration = 240  # Target duration of audio clips in seconds

        # Model architecture parameters
        self.input_size = self.n_mels  # Input feature dimension (Mel bins)
        self.hidden_size = 512  # LSTM hidden layer size
        self.num_layers = 2  # Number of LSTM layers
        self.bidirectional = True  # Whether to use a bidirectional LSTM
        self.output_size = 88  # Number of output nodes, corresponding to piano key range
        self.dropout = 0.3