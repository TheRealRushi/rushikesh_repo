import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
import sys
import time
import mediapipe as mp
from model_utils import (ACTIONS, SEQ_LENGTH, mp_holistic, mediapipe_detection, extract_keypoints)

mp_drawing = mp.solutions.drawing_utils


def create_ui():
    """
    Create a Tkinter window to display the latest prediction, its accuracy,
    and the sentence built so far from confirmed words.
    """
    root = tk.Tk()
    root.title("Sign Detection Info")
    root.geometry("400x250")

    label_action = tk.Label(root, text="Latest Prediction: ", font=("Helvetica", 16))
    label_accuracy = tk.Label(root, text="Accuracy: ", font=("Helvetica", 16))
    label_sentence = tk.Label(root, text="Selected Sentence: ", font=("Helvetica", 16), wraplength=380)

    label_action.pack(pady=5)
    label_accuracy.pack(pady=5)
    label_sentence.pack(pady=5)

    info = tk.Label(root, text="Instructions:\nPress 'c' to confirm word\nPress 's' to submit sentence\nPress 'q' to quit",
                    font=("Helvetica", 12))
    info.pack(pady=10)

    return root, label_action, label_accuracy, label_sentence


def translate_and_tts(sentence, target_language='en'):
    """
    Placeholder for translation and text-to-speech conversion.
    """
    print(f"Final Sentence: {sentence}")
    print(f"Translating and synthesizing speech for language: {target_language}")
    time.sleep(1)
    print("Speech synthesis completed.")


def real_time_detection(model):
    """
    Run real-time sign detection using a loaded model.
    Captures video frames via webcam, extracts keypoints for prediction,
    and shows prediction results in a separate Tkinter UI.
    """
    sequence = []
    threshold = 0.9  # Higher threshold for better accuracy
    selected_words = []  # Final confirmed words
    sentence_complete = False  # Flag to mark when sentence is complete

    ui_root, label_action, label_accuracy, label_sentence = create_ui()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        sys.exit()

    expected_seq_length = model.input_shape[1] if model.input_shape and model.input_shape[1] is not None else SEQ_LENGTH

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        confidence = 0
        latest_prediction = "Uncertain"

        while cap.isOpened() and not sentence_complete:
            ret, frame = cap.read()
            if not ret:
                continue

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)

            if keypoints is None:
                continue  # Skip frames where keypoints are missing

            sequence.append(keypoints)

            if len(sequence) == expected_seq_length:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                predicted_action = ACTIONS[predicted_idx]
                latest_prediction = predicted_action if confidence > threshold else "Uncertain"

                # Keep the most recent frames to maintain sequence length
                sequence = sequence[-(expected_seq_length - 1):]

                label_action.config(text=f"Latest Prediction: {latest_prediction}")
                label_accuracy.config(text=f"Accuracy: {confidence * 100:.1f}%")
                ui_root.update_idletasks()
                ui_root.update()

            cv2.imshow('Sign Language Detection', image)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                if confidence > threshold:
                    selected_words.append(latest_prediction)
                    label_sentence.config(text=f"Selected Sentence: {' '.join(selected_words)}")
                    print(f"Word confirmed: {latest_prediction}")
                    sequence.clear()  # Reset sequence to start fresh for next word
                else:
                    print("Prediction confidence too low to confirm.")
            elif key == ord('s'):
                if selected_words:
                    answer = messagebox.askyesno("Sentence Confirmation", "Is your sentence complete? (Yes to finish, No to continue)")
                    if answer:
                        final_sentence = ' '.join(selected_words)
                        translate_and_tts(final_sentence, target_language='en')
                        selected_words.clear()
                        label_sentence.config(text="Selected Sentence: ")
                        sentence_complete = True  # Stop detection after the sentence is confirmed

    cap.release()
    cv2.destroyAllWindows()
    ui_root.destroy()

if __name__ == "__main__":
    trained_model_path = r"C:\Users\The Baby\Rushikesh\isl_lstm\temp\model.keras"  # Update the path as needed.
    try:
        model = tf.keras.models.load_model(trained_model_path)
        print(f"Loaded trained model from {trained_model_path}")
    except Exception as e:
        print(f"Error loading trained model: {e}")
        sys.exit()

    real_time_detection(model)
