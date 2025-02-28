import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model_utils import ACTIONS, create_model

# Updated constant for the desired number of frames per sequence
TARGET_FRAMES = 45

# Update the data path to the new directory
DATA_PATH = 'training_data'

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom callback to save the model in the native Keras format (model.keras)
    when the validation loss improves.
    """
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss")
        if current_loss is None:
            return
        # If current validation loss improves, save the model.
        if current_loss < self.best:
            self.best = current_loss
            # Save the model using the native Keras format (no unsupported options passed)
            self.model.save(self.filepath, save_format="keras")
            print(f"\nEpoch {epoch + 1}: val_loss improved to {current_loss:.4f}, saving model.")

def train_model():
    """
    Train the LSTM model based on collected sequences.

    Loads sequences from session folders for each action and filters out sequences that don't have exactly TARGET_FRAMES frames.
    Splits the data and trains the model using callbacks for early stopping and learning rate reduction.
    Uses a custom checkpoint callback to save the model in native Keras format (model.keras).
    The model architecture is updated to accept sequences with shape (TARGET_FRAMES, features).
    """
    sequences, labels = [], []
    for action_idx, action in enumerate(ACTIONS):
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_dir):
            continue
        for session in os.listdir(action_dir):
            session_path = os.path.join(action_dir, session)
            if os.path.isdir(session_path):
                for seq_file in os.listdir(session_path):
                    if seq_file.endswith('.npy'):
                        seq_path = os.path.join(session_path, seq_file)
                        sequence = np.load(seq_path)
                        # Check if the sequence has exactly TARGET_FRAMES frames
                        if len(sequence) != TARGET_FRAMES:
                            print(f"Skipping {seq_path}: found {len(sequence)} frames instead of {TARGET_FRAMES}.")
                            continue
                        sequences.append(sequence)
                        labels.append(action_idx)
    if len(sequences) == 0:
        sys.exit("No valid training data found. Please run data collection first.")

    X = np.array(sequences)
    if X.ndim < 3:
        sys.exit("Data shape is incorrect. Ensure each sequence has TARGET_FRAMES frames with keypoint features.")
    # One-hot encoding with number of classes equal to len(ACTIONS)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(ACTIONS)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Create model with input shape (TARGET_FRAMES, features)
    model = create_model((TARGET_FRAMES, X.shape[2]), len(ACTIONS))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        CustomModelCheckpoint('model.keras'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=32,
        callbacks=callbacks
    )
    return model

if __name__ == "__main__":
    train_model()