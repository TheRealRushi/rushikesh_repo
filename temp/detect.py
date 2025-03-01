import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
import sys
import time
from model_utils import (ACTIONS, SEQ_LENGTH, mp_holistic, mediapipe_detection, extract_keypoints)
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

def create_ui():
    """
    Create a Tkinter window to display the latest prediction, its accuracy,
    and the sentence built so far from confirmed words.
    Returns the Tkinter root and label widgets.
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

    info = tk.Label(root, text="Instructions:\nPress 'c' to confirm\nPress 's' to submit sentence\nPress 'q' to quit",
                    font=("Helvetica", 12))
    info.pack(pady=10)

    return root, label_action, label_accuracy, label_sentence

def translate_and_tts(sentence, target_language='en'):
    """
    Placeholder for translation and text-to-speech conversion.
    """
    print("Final Sentence:", sentence)
    print("Translating and synthesizing speech for language:", target_language)
    time.sleep(1)
    print("Speech synthesis completed.")

def real_time_detection(model):
    """
    Run real-time sign detection using a loaded model.
    Captures video frames via webcam, extracts keypoints for prediction,
    and shows prediction results in a separate Tkinter UI.

    The user can confirm a prediction by pressing 'c', which adds the predicted
    word to a selected sentence. Press 's' to submit the sentence, triggering
    translation and text synthesis. The video window remains free of overlays.

    Press 'q' in the video window to exit.
    """
    sequence = []
    predictions = []
    threshold = 0.8  # Confidence threshold for predictions
    selected_words = []  # Final confirmed words

    ui_root, label_action, label_accuracy, label_sentence = create_ui()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        sys.exit()

    # Determine the expected sequence length from the loaded model.
    # This allows the detection code to adapt if the model expects a different length.
    expected_seq_length = model.input_shape[1] if model.input_shape and model.input_shape[1] is not None else SEQ_LENGTH

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        confidence = 0  # Initialize confidence variable.
        latest_prediction = "Uncertain"
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == expected_seq_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                predicted_action = ACTIONS[predicted_idx]

                latest_prediction = predicted_action if confidence > threshold else "Uncertain"
                # Keep the most recent frames to maintain the expected window size
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
                    print("Word confirmed:", latest_prediction)
                else:
                    print("Prediction confidence too low to confirm.")
            elif key == ord('s'):
                if selected_words:
                    final_sentence = ' '.join(selected_words)
                    translate_and_tts(final_sentence, target_language='en')
                    selected_words = []
                    label_sentence.config(text="Selected Sentence: ")

    cap.release()
    cv2.destroyAllWindows()
    ui_root.destroy()
    return

if __name__ == "__main__":
    # Instead of creating a dummy model, load your trained model.
    # For example, if you saved it as model.keras:
    trained_model_path = "C:/Users/Malhar/PycharmProjects/lstm/temp/model.keras"  # Update the path as needed.
    try:
        model = tf.keras.models.load_model(trained_model_path)
        print(f"Loaded trained model from {trained_model_path}")
    except Exception as e:
        print(f"Error loading trained model: {e}")
        sys.exit()

    real_time_detection(model)