import os
import cv2
import numpy as np
import sys
import time
import threading
import tensorflow as tf

# Import organization-specific modules and functions
from model_utils import ACTIONS, SEQ_LENGTH, NUM_SEQUENCES, mp_holistic, mp_drawing, mediapipe_detection, extract_keypoints

# Update the data path to the new directory
DATA_PATH = 'training_data'

# Set the desired number of frames per sequence to 45
TARGET_FRAMES = 45

# -----------------------
# GPU Acceleration Setup
# -----------------------
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
                current_frame = frame.copy()
        else:
            time.sleep(0.01)

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
    Perform simple data augmentation on a sequence by adding small Gaussian noise.
    """
    sequence_arr = np.array(sequence)
    noise = np.random.normal(0, noise_std, sequence_arr.shape)
    augmented = sequence_arr + noise
    return augmented

# -----------------------
# Main Data Collection Function
# -----------------------
def collect_data():
    """
    Collect data for each action (word) using a webcam.
    This version captures exactly TARGET_FRAMES (45) per sequence.

    Global controls during data collection:
      - Press 'd' to discard a sequence during capture.
      - Press 'p' to toggle pause.
      - Press 'q' to abort data collection.
    """
    global capture_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot access the camera.")

    # Start background camera thread
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
                    # Countdown before starting sequence capture
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

                    # Record a sequence with exactly 45 frames
                    recorded = False
                    sequence = []
                    paused = False
                    while len(sequence) < TARGET_FRAMES:
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

                            status_text = f"Session {session_count} | Seq {seq} | Frame {len(sequence)+1}/{TARGET_FRAMES}"
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
                            sequence = []  # Discard current sequence
                            break
                        elif key == ord('p'):
                            paused = not paused
                            if paused:
                                print("Data collection paused. Press 'p' to resume.")
                            else:
                                print("Data collection resumed.")

                    # Confirm and save sequence if exactly TARGET_FRAMES have been captured
                    if len(sequence) == TARGET_FRAMES:
                        confirm_image = image.copy()
                        cv2.putText(confirm_image, "Keep this sequence? (y=Yes, d=Discard)",
                                    (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collection', confirm_image)
                        print(f"Sequence {seq} captured. Press 'y' to save or 'd' to re-record.")
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
                                # Save augmented sequence copy
                                augmented_sequence = augment_sequence(sequence)
                                aug_path = os.path.join(session_dir, f"seq_{seq}_aug.npy")
                                np.save(aug_path, augmented_sequence)
                                print(f"Saved augmented sequence {seq} for '{action}'.")
                                seq += 1
                                break
                            elif key == ord('d'):
                                print(f"Re-recording sequence {seq} in session {session_count} for '{action}'.")
                                break
                    else:
                        print("Sequence not completed. Re-recording.")

                print(f"Session {session_count} for word '{action}' completed.")
                print("Press 'n' to record another session for this word, or any other key to move on.")
                key = cv2.waitKey(0) & 0xFF
                if key != ord('n'):
                    break

    capture_running = False
    thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
    # Uncomment the following lines to perform model quantization if needed.
    """
    saved_model_directory = '/path/to/saved_model'
    output_quant_model = '/path/to/quantized_model.tflite'
    quantize_model(saved_model_directory, output_quant_model)
    """