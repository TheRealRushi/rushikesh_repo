import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model_utils import (ACTIONS, DATA_PATH, SEQ_LENGTH,
                         create_model)


def train_model():
    """
    Train the LSTM model based on collected sequences.

    Loads sequences from session folders for each action and filters out sequences that don't have the required number of frames.
    Splits the data and trains the model using callbacks for early stopping, checkpointing, and learning rate reduction.
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
                        if len(sequence) != SEQ_LENGTH:
                            print(f"Skipping {seq_path}: found {len(sequence)} frames instead of {SEQ_LENGTH}.")
                            continue
                        sequences.append(sequence)
                        labels.append(action_idx)
    if len(sequences) == 0:
        sys.exit("No valid training data found. Please run data collection first.")
    X = np.array(sequences)
    if X.ndim < 3:
        sys.exit("Data shape is incorrect. Ensure each sequence has SEQ_LENGTH frames with keypoint features.")
    y = tf.keras.utils.to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = create_model((SEQ_LENGTH, X.shape[2]), len(ACTIONS))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True),
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
    trained_model = train_model()
    print("Training completed and model saved as 'model.h5'")