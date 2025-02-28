import os
import numpy as np

TARGET_FRAMES = 45


def resample_sequence(sequence, target_length=TARGET_FRAMES):
    """
    Resample a sequence (numpy array) to a fixed target length using uniform sampling.
    If the sequence has fewer than target_length frames, it pads by repeating the last frame.

    Args:
      sequence (np.ndarray): Input array with shape (num_frames, features)
      target_length (int): The desired number of frames (default: 45)

    Returns:
      np.ndarray: Resampled sequence with shape (target_length, features)
    """
    num_frames = len(sequence)
    # If already the correct length, return as-is.
    if num_frames == target_length:
        return sequence
    elif num_frames > target_length:
        # Uniformly sample indices
        indices = np.linspace(0, num_frames - 1, target_length, dtype=int)
        return sequence[indices]
    else:
        # If not enough frames, pad by repeating the last frame.
        pad_count = target_length - num_frames
        pad = np.repeat(sequence[-1][np.newaxis, :], pad_count, axis=0)
        return np.concatenate([sequence, pad], axis=0)


def process_npy_files(directory, target_length=TARGET_FRAMES):
    """
    Traverse the given directory recursively and resample .npy files to target_length frames.
    Files are overwritten with the resampled sequence.

    Args:
      directory (str): Root directory to search for .npy files.
      target_length (int): Desired frame count per sequence.

    Returns:
      None
    """
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith('.npy'):
                continue

            file_path = os.path.join(root, filename)
            try:
                # Load file; allow_pickle=True in case file was saved with pickling.
                sequence = np.load(file_path, allow_pickle=True)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            # Check if sequence has at least one dimension, then determine frame count.
            if sequence.ndim == 0:
                print(f"Skipping {file_path}: array has no dimensions.")
                continue

            # Assume frames are along axis 0.
            num_frames = len(sequence)
            if num_frames != target_length:
                resampled = resample_sequence(sequence, target_length)
                try:
                    np.save(file_path, resampled)
                    print(f"Resampled {file_path}: {num_frames} --> {target_length} frames.")
                except Exception as e:
                    print(f"Error saving {file_path}: {e}")
            else:
                print(f"No operation needed for {file_path}: already {target_length} frames.")


if __name__ == "__main__":
    # Specify the directory containing your .npy files.
    directory_path = "C:/Users/Malhar/PycharmProjects/lstm/temp/training_data"
    process_npy_files(directory_path)