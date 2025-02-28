import os
import cv2
import numpy as np
import sys
import time
import threading
import tensorflow as tf

# Import organization-specific modules and functions
from model_utils import (ACTIONS, DATA_PATH, SEQ_LENGTH, NUM_SEQUENCES,
                         mp_holistic, mp_drawing, mediapipe_detection, extract_keypoints)

# -----------------------
# GPU Acceleration Setup
# -----------------------
# Enable memory growth for GPUs to avoid allocation issues.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except Exception as e:
        print(f"Failed to set GPU memory growth: {e}")

# -----------------------
# Global Variables for Background Camera Frame Capture
# -----------------------
current_frame = None
frame_lock = threading.Lock()
capture_running = True  # Flag to signal the camera thread to stop


def camera_capture_thread(cap):
    """
    Background thread function to continuously capture frames from the camera.
    """
    global current_frame, capture_running
    while capture_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                # Store a copy of the frame to avoid race conditions.
                current_frame = frame.copy()
        else:
            time.sleep(0.01)  # Prevent busy waiting if read fails


def get_current_frame():
    """
    Safely retrieve the most recent frame captured by the background thread.
    """
    with frame_lock:
        return None if current_frame is None else current_frame.copy()


# -----------------------
# Data Augmentation Function
# -----------------------
def augment_sequence(sequence, noise_std=0.01):
    """
    Perform simple data augmentation on a sequence by adding
    small Gaussian noise to each keypoint array.

    Args:
        sequence (list or numpy array): The original sequence of keypoint arrays.
        noise_std (float): Standard deviation for the Gaussian noise.

    Returns:
        np.ndarray: Augmented sequence.
    """
    sequence_arr = np.array(sequence)
    noise = np.random.normal(0, noise_std, sequence_arr.shape)
    augmented = sequence_arr + noise
    return augmented


# -----------------------
# Optional: Quantized Model Generation
# -----------------------
def quantize_model(saved_model_dir, quantized_model_path):
    """
    Convert a SavedModel to a quantized TFLite model for faster inference.

    Args:
        saved_model_dir (str): Directory of the SavedModel.
        quantized_model_path (str): Output path for the quantized TFLite model.
    """
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        with open(quantized_model_path, 'wb') as f:
            f.write(tflite_quant_model)
        print(f"Quantized model saved to: {quantized_model_path}")
    except Exception as e:
        print(f"Model quantization failed: {e}")


# -----------------------
# Main Data Collection Function
# -----------------------
def collect_data():
    """
    Collect data for each action (word) using a webcam. This enhanced version includes:
      - Data augmentation.
      - Background threading for camera capture.
      - GPU acceleration support (if available).
      - An optional quantized model generation function.

    Modification:
      - Each sequence is now recorded for 10 seconds.

    Global controls during data collection:
      - Press 'd' to discard a sequence during capture.
      - Press 'p' to toggle pause.
      - Press 'q' to abort data collection.
    """
    global capture_running

    # Open camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot access the camera.")

    # Start background camera capture thread
    thread = threading.Thread(target=camera_capture_thread, args=(cap,), daemon=True)
    thread.start()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            action_dir = os.path.join(DATA_PATH, action)
            os.makedirs(action_dir, exist_ok=True)
            session_count = 0

            while True:
                session_count += 1
                session_dir = os.path.join(action_dir, f"session_{session_count}")
                os.makedirs(session_dir, exist_ok=True)
                print(f"\n--- Recording session {session_count} for word '{action}' ---")

                seq = 0
                while seq < NUM_SEQUENCES:
                    # Countdown before starting the sequence capture
                    for countdown in range(3, 0, -1):
                        frame = get_current_frame()
                        if frame is None:
                            continue
                        countdown_text = f"Get Ready: {countdown}"
                        text_size, _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
                        text_x = int((frame.shape[1] - text_size[0]) / 2)
                        cv2.putText(frame, countdown_text, (text_x, int(frame.shape[0] / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow('Data Collection', frame)
                        cv2.waitKey(1000)

                    # Record a sequence for 10 seconds
                    recorded = False
                    while not recorded:
                        sequence = []
                        paused = False
                        start_time = time.time()

                        while time.time() - start_time < 10:
                            frame = get_current_frame()
                            if frame is None:
                                continue

                            if not paused:
                                image, results = mediapipe_detection(frame, holistic)

                                # Draw landmarks if available
                                if results.pose_landmarks:
                                    mp_drawing.draw_landmarks(
                                        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                                if results.face_landmarks:
                                    mp_drawing.draw_landmarks(
                                        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                                if results.left_hand_landmarks:
                                    mp_drawing.draw_landmarks(
                                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                                if results.right_hand_landmarks:
                                    mp_drawing.draw_landmarks(
                                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                                status_text = f"Session {session_count} | Seq {seq} | Time: {int(time.time() - start_time)}s"
                                cv2.putText(image, status_text, (15, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                                keypoints = extract_keypoints(results)
                                sequence.append(keypoints)

                            cv2.imshow('Data Collection', image)
                            key = cv2.waitKey(33) & 0xFF

                            # Global controls during sequence capture
                            if key == ord('q'):
                                capture_running = False
                                cap.release()
                                cv2.destroyAllWindows()
                                sys.exit("Data collection aborted by user.")
                            elif key == ord('d'):
                                print(f"Discarded sequence during capture in session {session_count} for '{action}'.")
                                sequence = []  # Discard the current sequence and reattempt.
                                break
                            elif key == ord('p'):
                                paused = not paused
                                if paused:
                                    print("Data collection paused. Press 'p' to resume.")
                                else:
                                    print("Data collection resumed.")

                        # Verify if a valid sequence was captured (non-empty and lasting 10 seconds)
                        if sequence:
                            # Confirmation prompt with overlay text on final frame
                            confirm_image = image.copy()
                            cv2.putText(confirm_image, "Keep this sequence? (y=Yes, d=Discard)",
                                        (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Data Collection', confirm_image)
                            print(f"Sequence {seq} captured. Press 'y' to save or 'd' to re-record this sequence.")
                            while True:
                                key = cv2.waitKey(0) & 0xFF
                                if key == ord('q'):
                                    capture_running = False
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    sys.exit("Data collection aborted by user.")
                                elif key == ord('y'):
                                    seq_path = os.path.join(session_dir, f"seq_{seq}.npy")
                                    np.save(seq_path, sequence)
                                    print(f"Saved sequence {seq} in session {session_count} for '{action}'.")
                                    # Save an augmented copy of the sequence
                                    augmented_sequence = augment_sequence(sequence)
                                    aug_path = os.path.join(session_dir, f"seq_{seq}_aug.npy")
                                    np.save(aug_path, augmented_sequence)
                                    print(f"Saved augmented sequence {seq} in session {session_count} for '{action}'.")
                                    recorded = True
                                    seq += 1
                                    break
                                elif key == ord('d'):
                                    print(f"Re-recording sequence {seq} in session {session_count} for '{action}'.")
                                    break
                        else:
                            print("Sequence was not completed. Re-recording the sequence.")

                print(f"Session {session_count} for word '{action}' completed.")
                print("Press 'n' to record another session for this word, or any other key to move to the next word.")
                key = cv2.waitKey(0) & 0xFF
                if key != ord('n'):
                    break

    # Stop background frame capture thread and release camera resources.
    capture_running = False
    thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_data()
    # Example: Uncomment the following lines to perform model quantization.
    """
    saved_model_directory = '/path/to/saved_model'
    output_quant_model = '/path/to/quantized_model.tflite'
    quantize_model(saved_model_directory, output_quant_model)
    """