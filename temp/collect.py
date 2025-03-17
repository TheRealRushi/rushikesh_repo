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
# Data Collection for a Single Action
# -----------------------
def record_action(action, cap, holistic):
    """
    Record data sequences for a single action (word) using the provided camera and holistic model.

    The function manages session creation, sequence capture, and interactive commands.

    Returns when user finishes or skips recording for the action.
    """
    print(f"\nReady to record data for word '{action}'.")
    print("Press 's' at any time to skip this word.")

    current_session_dir = None
    session_count = 0
    current_seq = 0  # Counter for sequences in the current session

    while True:
        # Create a new session if none exists or if the current session is full.
        if current_session_dir is None or current_seq >= NUM_SEQUENCES:
            if current_seq >= NUM_SEQUENCES:
                print(f"Current session '{current_session_dir}' is full (max {NUM_SEQUENCES} sequences).")
            session_count += 1
            current_session_dir = os.path.join(DATA_PATH, action, f"session_{session_count}")
            os.makedirs(current_session_dir, exist_ok=True)
            current_seq = 0
            print(f"\n--- Starting session {session_count} for word '{action}' ---")

        # Capture sequences for the current session until the user makes a decision.
        while current_seq < NUM_SEQUENCES:
            skip_current_word = False
            # Countdown before starting sequence capture.
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
                key = cv2.waitKey(1000) & 0xFF
                if key == ord('s'):
                    print(f"Skip command received. Skipping word '{action}'.")
                    skip_current_word = True
                    break
            if skip_current_word:
                break

            sequence = []
            paused = False
            # Record exactly TARGET_FRAMES frames
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

                    status_text = f"Session {session_count} | Seq {current_seq} | Frame {len(sequence) + 1}/{TARGET_FRAMES}"
                    cv2.putText(image, status_text, (15, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)

                cv2.imshow('Data Collection', image)
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    # Abort the entire data collection process.
                    return False
                elif key == ord('d'):
                    print(f"Discarded sequence during capture in session {session_count} for '{action}'.")
                    sequence = []  # Discard current sequence and restart it
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused." if paused else "Resumed.")
                elif key == ord('s'):
                    print(f"Skip command received. Skipping word '{action}'.")
                    skip_current_word = True
                    break
            if skip_current_word:
                break

            # Confirm and save sequence if exactly TARGET_FRAMES have been captured.
            if len(sequence) == TARGET_FRAMES:
                confirm_image = image.copy()
                cv2.putText(confirm_image, "Keep this sequence? (y=Yes, d=Discard)",
                            (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', confirm_image)
                print(f"Sequence {current_seq} captured in session {session_count} for '{action}'.")
                print("Press 'y' to save, 'd' to re-record, or 's' to skip this word.")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        return False
                    elif key == ord('y'):
                        seq_path = os.path.join(current_session_dir, f"seq_{current_seq}.npy")
                        np.save(seq_path, sequence)
                        print(f"Saved sequence {current_seq} in session {session_count} for '{action}'.")
                        # Save augmented sequence copy.
                        augmented_sequence = augment_sequence(sequence)
                        aug_path = os.path.join(current_session_dir, f"seq_{current_seq}_aug.npy")
                        np.save(aug_path, augmented_sequence)
                        print(f"Saved augmented sequence {current_seq} for '{action}'.")
                        current_seq += 1
                        break
                    elif key == ord('d'):
                        print(f"Re-recording sequence {current_seq} in session {session_count} for '{action}'.")
                        break
                    elif key == ord('s'):
                        print(f"Skip command received. Skipping word '{action}'.")
                        skip_current_word = True
                        break
                if skip_current_word:
                    break
            else:
                print("Sequence not completed. Re-recording.")
                continue  # re-record current sequence

        if skip_current_word:
            print(f"Skipping further recording for word '{action}'.")
            break

        # Prompt: update current session or start a new session?
        print(f"\nSession {session_count} for word '{action}' completed.")
        print("Press 'u' to update (add more sequences to this session),")
        print("Press 'n' to start a new session,")
        print("Press 's' to skip this word,")
        print("or any other key to finish recording for '{action}'.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('u'):
            if current_seq >= NUM_SEQUENCES:
                print("Current session is full. Starting a new session.")
                current_session_dir = None  # force new session creation
            else:
                print("Updating current session.")
                continue
        elif key == ord('n'):
            print("Starting a new session for this word.")
            current_session_dir = None  # force new session creation
            continue
        elif key == ord('s'):
            print(f"Skipping further recording for word '{action}'.")
            break
        else:
            print(f"Finished recording for word '{action}'.")
            break

    # Return True to indicate successful completion for this action.
    return True

# -----------------------
# Main Data Collection Function with Interactive Word Loop
# -----------------------
def collect_data():
    global capture_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot access the camera.")

    # Start background camera thread
    thread = threading.Thread(target=camera_capture_thread, args=(cap,), daemon=True)
    thread.start()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Outer loop: repeatedly prompt for action selection.
        while True:
            print("\nAvailable actions:")
            for action in ACTIONS:
                print(f"  - {action}")
            selected = input("Enter the action you want to record (or type 'exit' to quit): ").strip()
            if selected.lower() == 'exit':
                break
            elif selected not in ACTIONS:
                print("Invalid action selected. Please try again.")
                continue

            # Record data for the selected action.
            success = record_action(selected, cap, holistic)
            if not success:
                # If record_action returns False, it signals an abort.
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