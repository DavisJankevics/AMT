import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from tensorflow.keras import regularizers

# class Attention(nn.Module):
#     def __init__(self, hidden_dim, bidirectional=True):
#         super(Attention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.multiplier = 4 if bidirectional else 2
#         self.attn = nn.Linear(self.hidden_dim * self.multiplier, self.hidden_dim)
#         self.v = nn.Parameter(torch.rand(hidden_dim))

#     def forward(self, hidden, encoder_outputs):
#         timestep = encoder_outputs.size(1)
#         print(hidden.size())
#         h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
#         print(encoder_outputs.size())
#         encoder_outputs = encoder_outputs.contiguous().view(-1, self.hidden_dim * 2)
#         print(encoder_outputs.size())
#         print(h.size())

#         attn_energies = self.score(h, encoder_outputs)
#         return F.softmax(attn_energies, dim=1).view(-1, timestep, 1)

#     def score(self, hidden, encoder_outputs):
#         print("encoder1")
#         print(encoder_outputs.size())
#         print("hidden1")
#         print(hidden.size())
#         # Ensure hidden is 3D: [batch_size, 1, num_directions * hidden_size]
#         if hidden.dim() == 2:
#             hidden = hidden.unsqueeze(1)
#         print("hidden2")

#         print(hidden.size())
#         # Encoder_outputs should be: [batch_size, seq_len, num_directions * hidden_size]
#         # If it's not, you need to verify why and adjust accordingly.
#         # For demonstration, assuming it needs to be reshaped:
#         if encoder_outputs.dim() == 2:
#             # This reshape is hypothetical and might not fit your exact case
#             # You need to know the correct seq_len and ensure the total elements match
#             seq_len = hidden.size(1)  # Or however seq_len should be determined
#             encoder_outputs = encoder_outputs.view(hidden.size(0), seq_len, -1)
#         print("encoder2")
#         print(encoder_outputs.size())
        
#         print(hidden.size())
#         print(torch.cat([hidden, encoder_outputs], dim=2).size())
#         # Now both tensors should be 3D, and you can proceed with concatenation and scoring
#         energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
#         return energy.squeeze(1)

# class BiLSTMAttention(nn.Module):
#     def __init__(self, config):
#         super(BiLSTMAttention, self).__init__()
#         self.bidirectional = config.bidirectional
#         self.lstm = nn.LSTM(
#             input_size=config.input_size,
#             hidden_size=config.hidden_size * 2,
#             num_layers=config.num_layers,
#             bidirectional=config.bidirectional,
#             batch_first=True,
#             dropout=config.dropout if config.num_layers > 1 else 0,
#         )
#         self.dropout = nn.Dropout(config.dropout)
#         self.attention = Attention(config.hidden_size, config.bidirectional)
#         self.fc = nn.Linear(
#             config.hidden_size * 2,
#             config.output_size
#         )
# # * (2 if config.bidirectional else 1)
#     def forward(self, x, lengths):
#         lengths_cpu = lengths.to('cpu')
#         packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
#         packed_output, (hidden, cell) = self.lstm(packed_x)
#         output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
#         # print() hidden[-1] size 
#         attn_weights = self.attention(hidden[-1] if self.bidirectional else hidden[0], output)
#         output = output * attn_weights
#         output = self.dropout(output)
#         output = self.fc(output)
#         # Apply softmax to the output layer to get probabilities
#         return F.softmax(output, dim=2)


# class BiLSTM(nn.Module):
#     def __init__(self, config):
#         super(BiLSTM, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=config.input_size,  # Number of MFCC features
#             hidden_size=config.hidden_size,  # LSTM hidden layer size
#             num_layers=config.num_layers,  # Number of LSTM layers
#             bidirectional=config.bidirectional,  # Use bidirectional LSTM
#             batch_first=True  # Input and output tensors are provided as (batch, seq, feature)
#         )
#         # The output layer that maps from hidden state space to class space
#         self.fc = nn.Linear(
#             config.hidden_size * 2 if config.bidirectional else config.hidden_size,
#             config.output_size  # Number of output classes (e.g., MIDI note numbers)
#         )

#     def forward(self, x, lengths):
#         # Pack the padded sequence
#         lengths_cpu = lengths.to('cpu')

#         packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
#         packed_output, (hidden, cell) = self.lstm(packed_x)
        
#         # Unpack the sequence
#         output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        
#         # Apply the fully connected layer to each time step
#         output = self.fc(output)
        
#         return output

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CombinePredictionsLayer(layers.Layer):
    def __init__(self, threshold=0.5, **kwargs):
        super(CombinePredictionsLayer, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, pitch_prediction, onset_prediction):
        # Assume pitch_prediction and onset_prediction are binary or near-binary (e.g., after applying a sigmoid)
        combined_prediction = pitch_prediction * onset_prediction
        # Thresholding to obtain a binary piano roll
        # piano_roll = tf.where(combined_prediction > self.threshold, 1.0, 0.0)
        return combined_prediction

    def get_config(self):
        config = super(CombinePredictionsLayer, self).get_config()
        config.update({"threshold": self.threshold})
        return config

class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def build_model(input_shape, num_notes, config):
    sequence_input = layers.Input(shape=input_shape, dtype='float32')

    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(sequence_input)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(x)
    # First BiLSTM layer
    lstm_out_1 = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, dropout=config.dropout, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))(x)
    lstm_out_1 = layers.add([lstm_out_1, x])

    # Apply attention layer for pitch detection
    pitch_attention = AttentionLayer(config.hidden_size * 2)
    pitch_context_vector, _ = pitch_attention(lstm_out_1, lstm_out_1)
    pitch_output = layers.Dense(num_notes, activation='sigmoid', dtype='float32')(pitch_context_vector)

    lstm_out_2 = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, dropout=config.dropout, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), dtype='float32'))(pitch_output)
    lstm_out_2 = layers.concatenate([lstm_out_2, pitch_output])
    # Apply attention layer for onset detection
    onset_attention = AttentionLayer(config.hidden_size * 2)
    onset_context_vector, _ = onset_attention(lstm_out_2, lstm_out_2)
    onset_output = layers.Dense(num_notes, activation='sigmoid', dtype='float32')(onset_context_vector)

    # Combining pitch and onset predictions
    # combined_input = layers.Concatenate(axis=-1)([pitch_output, onset_output])

    # Second BiLSTM layer receives combined context vectors from pitch and onset detections
    lstm_out_3 = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, dropout=config.dropout, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), dtype='float32'))(onset_output)
    lstm_out_3 = layers.concatenate([lstm_out_3, pitch_output])
    lstm_out_3 = layers.concatenate([lstm_out_3, onset_output])

    # Final output dense layer for the prediction
    output = layers.TimeDistributed(layers.Dense(num_notes, activation='sigmoid', dtype='float32'))(lstm_out_3)

    model = keras.Model(inputs=sequence_input, outputs=output)
    model.summary()
    return model


# def build_model(input_shape, num_notes, config):

#     sequence_input = keras.Input(shape=input_shape, dtype='float32')
#     # LSTM layers with return_sequences=True to output sequences for attention layer
#     lstm_out = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, dropout=config.dropout if config.num_layers > 1 else 0, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), dtype='float32'))(sequence_input)
#     for _ in range(1, config.num_layers):
#         lstm_out = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, dropout=config.dropout, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), dtype='float32'))(lstm_out)
#     # Applying attention to each timestep in the LSTM output sequence
#     attention = AttentionLayer(config.hidden_size * 2)
#     context_vector, attention_weights = attention(lstm_out, lstm_out)

#     # Dense layer processing
#     dense_out = layers.Dense(config.hidden_size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), dtype='float32')(context_vector)
#     dropout_out = layers.Dropout(config.dropout, dtype='float32')(dense_out)

#     # Time-distributed output layer for time-step-wise prediction
#     output = layers.TimeDistributed(layers.Dense(num_notes, activation='softmax', dtype='float32'))(dropout_out)

#     model = keras.Model(inputs=sequence_input, outputs=output)
#     model.summary()
#     return model

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# class AttentionLayer(layers.Layer):
#     def __init__(self, units):
#         super(AttentionLayer, self).__init__()
#         self.W1 = layers.Dense(units)
#         self.W2 = layers.Dense(units)
#         self.V = layers.Dense(1)

#     def call(self, query, values):
#         query_with_time_axis = tf.expand_dims(query, 1)
#         score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
#         attention_weights = tf.nn.softmax(score, axis=1)
#         context_vector = attention_weights * values
#         context_vector = tf.reduce_sum(context_vector, axis=1)
#         return context_vector, attention_weights

# def build_model(input_shape, num_notes, config):

#     sequence_input = keras.Input(shape=input_shape, dtype='float32')
#     # LSTM layers with return_sequences=True to output sequences for attention layer
#     lstm_out = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, dropout=config.dropout if config.num_layers > 1 else 0, dtype='float32'))(sequence_input)
#     attention = AttentionLayer(config.hidden_size * 2)
#     context_vector, attention_weights = attention(lstm_out, lstm_out)

#     # Dense layer processing
#     dense_out = layers.Dense(config.hidden_size, activation='relu', dtype='float32')(context_vector)
#     for _ in range(1, config.num_layers):
#         lstm_out = layers.Bidirectional(layers.LSTM(config.hidden_size, return_sequences=True, dropout=config.dropout, dtype='float32'))(dense_out)
#         attention = AttentionLayer(config.hidden_size * 2)
#         context_vector, attention_weights = attention(lstm_out, lstm_out)

#         # Dense layer processing
#         dense_out = layers.Dense(config.hidden_size, activation='relu', dtype='float32')(context_vector)
#     # Applying attention to each timestep in the LSTM output sequence
#     # attention = AttentionLayer(config.hidden_size * 2)
#     # context_vector, attention_weights = attention(lstm_out, lstm_out)

#     # # Dense layer processing
#     # dense_out = layers.Dense(config.hidden_size, activation='relu', dtype='float32')(context_vector)
#     dropout_out = layers.Dropout(config.dropout, dtype='float32')(dense_out)

#     # Time-distributed output layer for time-step-wise prediction
#     output = layers.TimeDistributed(layers.Dense(num_notes, activation='softmax', dtype='float32'))(dropout_out)

#     model = keras.Model(inputs=sequence_input, outputs=output)
#     model.summary()
#     return model