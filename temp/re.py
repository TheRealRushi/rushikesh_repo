import os
import numpy as np


def resample_sequence(sequence, target_length):
    """
    Resample a sequence (numpy array of frames) to a fixed target length
    using uniform sampling.

    Args:
      sequence (np.ndarray): Array with shape (num_frames, features)
      target_length (int): Desired frame count for the output sequence

    Returns:
      np.ndarray: Resampled sequence with shape (target_length, features)
    """
    current_length = len(sequence)
    # If the sequence already has the target number, return it unchanged.
    if current_length == target_length:
        return sequence
    elif current_length > target_length:
        # Uniformly sample indices from the existing sequence
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return sequence[indices]
    else:
        # If there are too few frames, we can either pad or skip.
        # Here we choose to pad by repeating the last frame.
        pad_count = target_length - current_length
        pad = np.repeat(sequence[-1][np.newaxis, :], pad_count, axis=0)
        return np.concatenate([sequence, pad], axis=0)


def process_i_word_data(data_path, target_length=45):
    """
    Process all .npy files in the provided data_path for the 'I' word.
    If a sequence does not have exactly target_length frames, it is resampled.
    The resampled sequences are saved with a suffix '_resampled.npy'.

    Args:
      data_path (str): Path to the directory containing the 'I' word session folders.
      target_length (int): Desired frame count per sequence.
    """
    # Walk through the directory for the I word.
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                try:
                    sequence = np.load(file_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                current_length = len(sequence)
                if current_length != target_length:
                    # Resample the sequence to exactly target_length frames.
                    resampled = resample_sequence(sequence, target_length)
                    new_file_path = file_path.replace('.npy', '_resampled.npy')
                    np.save(new_file_path, resampled)
                    print(
                        f"Resampled {file_path}: {current_length} -> {target_length} frames (saved as {new_file_path}).")
                else:
                    print(f"Sequence {file_path} already has {target_length} frames.")


if __name__ == "__main__":
    # Update to the directory of your collected 'I' word data.
    # For example, if your data is stored under:
    # "../Realtime-Sign-Language-Detection-Using-LSTM-Model/MP_Data/I"
    DATA_PATH = "../Realtime-Sign-Language-Detection-Using-LSTM-Model/MP_Data/I"
    process_i_word_data(DATA_PATH, target_length=45)
    print("Resampling process completed.")