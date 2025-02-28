import os
import cv2
import numpy as np
import sys
import time
from model_utils import ACTIONS, DATA_PATH, SEQ_LENGTH, NUM_SEQUENCES, mp_holistic, mp_drawing, mediapipe_detection, extract_keypoints

def collect_data():
    """
    Collect data for each action (word) using a webcam with discard, pause, and confirmation functionality.

    Features added:
      - Discard current sequence during frame collection with 'd'.
      - Pause/resume data collection with 'p'.
      - After a sequence is complete, prompt for confirmation to keep (press 'y') or discard (press 'd').
      - Adjust frame collection for 30 FPS camera.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot access the camera.")

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

                # Iterate until NUM_SEQUENCES sequences are successfully captured and confirmed
                seq = 0
                while seq < NUM_SEQUENCES:
                    # Countdown (3-second countdown)
                    for countdown in range(3, 0, -1):
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        count_text = f"Get Ready: {countdown}"
                        count_text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
                        count_text_x = int((frame.shape[1] - count_text_size[0]) / 2)
                        cv2.putText(frame, count_text, (count_text_x, int(frame.shape[0] / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow('Data Collection', frame)
                        cv2.waitKey(1000)

                    # Capture a sequence until confirmation is received
                    recorded = False
                    while not recorded:
                        sequence = []
                        frame_num = 0
                        paused = False

                        while frame_num < SEQ_LENGTH:
                            ret, frame = cap.read()
                            if not ret:
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

                                # Display current action and status information
                                word_text = f"Current Word: {action}"
                                word_text_size, _ = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                                word_text_x = int((image.shape[1] - word_text_size[0]) / 2)
                                cv2.putText(image, word_text, (word_text_x, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

                                status_text = f"Session {session_count} | Seq {seq} | Frame {frame_num}"
                                cv2.putText(image, status_text, (15, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                                if frame_num == 0:
                                    cv2.putText(image, "STARTING COLLECTION", (120, 200),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                                keypoints = extract_keypoints(results)
                                sequence.append(keypoints)
                                frame_num += 1

                            cv2.imshow('Data Collection', image)
                            key = cv2.waitKey(33) & 0xFF  # Adjusted for 30 FPS

                            # Global controls during sequence capture
                            if key == ord('q'):
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

                        # Only proceed if a full sequence was captured
                        if len(sequence) == SEQ_LENGTH:
                            # Confirmation prompt: show final frame with confirmation text overlay
                            confirm_image = image.copy()
                            cv2.putText(confirm_image, "Keep this sequence? (y=Yes, d=Discard)",
                                        (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Data Collection', confirm_image)
                            print(f"Sequence {seq} captured. Press 'y' to save or 'd' to re-record this sequence.")
                            while True:
                                key = cv2.waitKey(0) & 0xFF
                                if key == ord('q'):
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    sys.exit("Data collection aborted by user.")
                                elif key == ord('y'):
                                    np.save(os.path.join(session_dir, f"seq_{seq}.npy"), sequence)
                                    print(f"Saved sequence {seq} in session {session_count} for '{action}'.")
                                    recorded = True
                                    seq += 1
                                    break
                                elif key == ord('d'):
                                    print(f"Re-recording sequence {seq} in session {session_count} for '{action}'.")
                                    # Break out of confirmation loop and re-record the same sequence
                                    break
                        else:
                            # If sequence was discarded during capture, reattempt capturing that sequence.
                            print("Sequence was not completed. Re-recording the sequence.")

                print(f"Session {session_count} for word '{action}' completed.")
                print("Press 'n' to record another session for this word, or any other key to move to the next word.")
                key = cv2.waitKey(0) & 0xFF
                if key != ord('n'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()