import argparse
import torch
from torch.utils.data import DataLoader
from model.model import build_model, AttentionLayer, CombinePredictionsLayer
from conf.conf import Config
from data.data import create_tf_dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, F1Score
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# def scheduler(epoch, lr):
#     if epoch%10 != 0:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        current_lr = lr.numpy()  # Convert the learning rate tensor to a numpy array
        print(f"\nEpoch {epoch+1}: Learning rate is {current_lr:.6f}.")

# def focal_loss(gamma=2., alpha=4.):
#     def focal_loss_fixed(y_true, y_pred):
#         """Focal loss for multi-classification
#         FL(p_t)=-alpha*(1-p_t)^gamma*log(p_t)
#         """
#         epsilon = tf.keras.backend.epsilon()
#         y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
#         cross_entropy = -y_true * tf.math.log(y_pred)
#         loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
#         return tf.reduce_sum(loss, axis=-1)
#     return focal_loss_fixed

def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/abs/1708.02002
    Usage:
     model.compile(optimizer='adam', loss=binary_focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])
    """
    """
    Binary form of focal loss with added stability in log computations.
    """
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()  # Small constant to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Clip predictions to avoid log(0) and log(1)

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        # Compute the focal loss for positive and negative samples separately
        loss_1 = -alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)
        loss_0 = -(1-alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)

        # Combine the losses (mean over the batch)
        loss = tf.reduce_mean(loss_1 + loss_0)

        return loss

    return focal_loss

class BatchMetricsLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        print(f"")
        
    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        print(f"\nValidation Batch {batch}, Loss: {logs.get('loss')}, Accuracy: {logs.get('binary_accuracy')}, Precision: {logs.get('precision')}, Recall: {logs.get('recall')}")

def train(db_location, load_model_path=None):
    config = Config()
    input_shape = (None, config.input_size)
    num_notes = config.output_size

    model = build_model(input_shape, num_notes, config)
    initial_epoch = 0
    
    if load_model_path:
        # Check if the path is a .ckpt file (weights only)
        filename = load_model_path.split('/')[-1]  # Get the file name
        epoch_str = filename.split('_')[-1]  # Split by "_" and get the epoch part
        initial_epoch = int(epoch_str.split('.')[0])
        print(f"Loading model from {load_model_path} at epoch {initial_epoch}.")
        if load_model_path.endswith('.h5'):
            # For .h5 files, you can load the full model (uncomment below line if needed)
            #   'binary_focal_loss': binary_focal_loss(gamma=2.,alpha=0.25)
            model = tf.keras.models.load_model(load_model_path, custom_objects={'AttentionLayer': AttentionLayer})
            print(f"Model loaded successfully from {load_model_path}.")
        else:
            # Load weights into the model
            model.load_weights(load_model_path)
            # accuracy = BinaryAccuracy(name = 'binary_accuracy', threshold = 0.5)
            # loss_function = BinaryFocalCrossentropy(gamma=config.gamma,alpha=config.alpha, apply_class_balancing=True)
            # optimizer = Adam(learning_rate=config.learning_rate)
            # model.compile(loss = loss_function, metrics=[accuracy, Precision(thresholds = 0.5), Recall(thresholds = 0.5)])
            print(f"Weights loaded successfully from {load_model_path}.")
    else:
        print("Starting training with a new model.")
        optimizer = Adam()
        # optimizer = Adam(learning_rate=config.learning_rate)
        loss_function = BinaryFocalCrossentropy(gamma=config.gamma,alpha=config.alpha, apply_class_balancing=True)
        accuracy = BinaryAccuracy(name = 'binary_accuracy', threshold = 0.5)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy, Precision(thresholds = 0.5), Recall(thresholds = 0.5)])

    # optimizer = Adam()
    # loss_function = BinaryFocalCrossentropy(gamma=3.,alpha=0.85, apply_class_balancing=True)
    # accuracy = BinaryAccuracy(name = 'binary_accuracy', threshold = 0.25)
    # model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy, Precision(thresholds = 0.5), Recall(thresholds = 0.5)])
    # train_dataset = create_tf_dataset(root_dir=db_location, split='train', sr=config.sr, hop_length=config.hop_length, n_mfcc=config.n_mfcc)
    # val_dataset = create_tf_dataset(root_dir=db_location, split='validation', sr=config.sr, hop_length=config.hop_length, n_mfcc=config.n_mfcc)
    train_dataset = create_tf_dataset(root_dir=db_location, split='train', sr=config.sr, hop_length=config.hop_length, n_fft=config.n_fft, n_mels=config.n_mels, target_duration=config.target_duration)
    val_dataset = create_tf_dataset(root_dir=db_location, split='validation', sr=config.sr, hop_length=config.hop_length, n_fft=config.n_fft, n_mels=config.n_mels, target_duration=config.target_duration)
    alpha, gamma = (config.alpha % 1) * 100, config.gamma
    callbacks = [
        ModelCheckpoint("/content/drive/MyDrive/model_mel_g3_a70_{epoch:03d}.h5", save_weights_only=False, save_best_only=False, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, min_delta=0, restore_best_weights=True, verbose=1, mode='auto'),
        BatchMetricsLogger(),
        LearningRateLogger()
    ]
    history = model.fit(
        train_dataset.batch(config.batch_size),  # Ensure batching is applied
        epochs=config.num_epochs,
        validation_data=val_dataset.batch(config.batch_size),  # Ensure batching is applied
        callbacks=[callbacks],
        initial_epoch=initial_epoch
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
