import argparse
import torch
from torch.utils.data import DataLoader
from model.model import build_model, AttentionLayer
from conf.conf import Config
from data.data import create_tf_dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# def load_model_checkpoint(checkpoint_path):
#     print("\n\n\n",checkpoint_path)
    
#     try:
#         model = tf.keras.models.load_model(checkpoint_path)
#         print(f"Model loaded successfully from {checkpoint_path}.")
#         return model
#     except Exception as e:
#         print(f"Error loading model from {checkpoint_path}: {e}")
#         return None

# class BatchMetricsCallback(Callback):
#     def __init__(self):
#         super(BatchMetricsCallback, self).__init__()
#         self.accuracy = tf.keras.metrics.Accuracy()
#         self.precision = tf.keras.metrics.Precision()
#         self.recall = tf.keras.metrics.Recall()

#     def on_train_batch_end(self, batch, logs=None):
#         logs = logs or {}
#         acc = logs.get('accuracy')
#         prec = logs.get('precision')
#         rec = logs.get('recall')
#         print("acc",acc)
#         print("prec",prec)
#         print("rec",rec)
        # if(acc is not None):
        #     f1_score = 2 * (prec * rec) / (prec + rec + tf.keras.backend.epsilon())
        #     print(f"\nBatch {batch}, Loss: {logs.get('loss'):.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1_score:.4f}")

def train(db_location, load_model_path=None):
    config = Config()
    input_shape = (None, config.input_size)
    num_notes = config.output_size
    # model = build_model(input_shape, num_notes, config)

    model = build_model(input_shape, num_notes, config)
    
    if load_model_path:
        # Check if the path is a .ckpt file (weights only)
        if load_model_path.endswith('.h5'):
            # For .h5 files, you can load the full model (uncomment below line if needed)
            model = tf.keras.models.load_model(load_model_path, custom_objects={'AttentionLayer': AttentionLayer})
            print(f"Model loaded successfully from {load_model_path}.")
        else:
            # Load weights into the model
            model.load_weights(load_model_path)
            print(f"Weights loaded successfully from {load_model_path}.")
    else:
        print("Starting training with a new model.")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_dataset = create_tf_dataset(root_dir=db_location, split='train', sr=config.sr, hop_length=config.hop_length, n_mfcc=config.n_mfcc)
    val_dataset = create_tf_dataset(root_dir=db_location, split='validation', sr=config.sr, hop_length=config.hop_length, n_mfcc=config.n_mfcc)
    
    callbacks = [
        ModelCheckpoint("./checkpoints/model_{epoch:03d}.h5", save_weights_only=False, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    history = model.fit(
        train_dataset.batch(config.batch_size),  # Ensure batching is applied
        epochs=config.num_epochs,
        validation_data=val_dataset.batch(config.batch_size),  # Ensure batching is applied
        callbacks=[callbacks]
    )
    model.save("./final_model.h5")
    model.save_weights("./final_model_w.ckpt")

    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BiLSTM for Music Transcription')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load the model checkpoint')
    args = parser.parse_args()

    train(args.db_location, args.load_model_path)
