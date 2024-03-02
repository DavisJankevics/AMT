class Config:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 64
        self.learning_rate = 0.001
        self.input_size = 2  # number of channels
        self.hidden_size = 256  # LSTM hidden size
        self.num_layers = 2  # number of LSTM layers
        self.bidirectional = True  # Bi-LSTM
