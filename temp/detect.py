import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
import sys
import time
from model_utils import (ACTIONS, SEQ_LENGTH, mp_holistic, mediapipe_detection, extract_keypoints)

# For drawing landmarks on video feed (optional)
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
    Intended to use external tools like coqui and sarvam.
    For demonstration, it prints the sentence and target language.
    """
    # Here you would integrate with the translation API (e.g., Sarvam)
    # and then pass the translated text to a TTS engine (e.g., Coqui TTS).
    print("Final Sentence:", sentence)
    print("Translating and synthesizing speech for language:", target_language)
    # Dummy behavior: just simulate delay.
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
    sentence = []  # Stores temporary sentence words for confirmation
    predictions = []
    threshold = 0.8  # Confidence threshold for predictions
    selected_words = []  # Final confirmed words

    # Initialize the UI for displaying prediction results and sentence
    ui_root, label_action, label_accuracy, label_sentence = create_ui()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        sys.exit()

    # Use a holistic detector outside the loop for efficiency
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Process frame and extract keypoints
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # When sequence is complete, run model prediction
            if len(sequence) == SEQ_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                predicted_action = ACTIONS[predicted_idx]

                # Update prediction if confidence threshold met
                if confidence > threshold:
                    # Keep track of latest prediction for confirmation
                    latest_prediction = predicted_action
                else:
                    latest_prediction = "Uncertain"

                # Reset sequence (using sliding window)
                sequence = sequence[-(SEQ_LENGTH - 1):]

                # Update Tkinter UI labels with the latest prediction and accuracy
                label_action.config(text=f"Latest Prediction: {latest_prediction}")
                label_accuracy.config(text=f"Accuracy: {confidence * 100:.1f}%")
                ui_root.update_idletasks()
                ui_root.update()

            # Display video feed without overlays
            cv2.imshow('Sign Language Detection', image)
            key = cv2.waitKey(10) & 0xFF

            # Global controls from the video window:
            # 'q' to quit, 'c' to confirm the current prediction,
            # 's' to submit the sentence.
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Confirm the latest prediction if its confidence is sufficient
                if confidence > threshold:
                    selected_words.append(latest_prediction)
                    # Update the sentence label in the UI
                    label_sentence.config(text=f"Selected Sentence: {' '.join(selected_words)}")
                    print("Word confirmed:", latest_prediction)
                else:
                    print("Prediction confidence too low to confirm.")
            elif key == ord('s'):
                # Finalize sentence and invoke translation and TTS
                if selected_words:
                    final_sentence = ' '.join(selected_words)
                    translate_and_tts(final_sentence, target_language='en')
                    # Reset selected words after submission
                    selected_words = []
                    label_sentence.config(text="Selected Sentence: ")

        # End of loop: release camera and destroy OpenCV and Tkinter windows
        cap.release()
        cv2.destroyAllWindows()
        ui_root.destroy()


if __name__ == "__main__":
    # Create a dummy model with an input shape matching our keypoints output.
    # Using SEQ_LENGTH timesteps with 1662 features per timestep.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    dummy_model = Sequential([
        Flatten(input_shape=(SEQ_LENGTH, 1662)),
        Dense(64, activation='relu'),
        Dense(len(ACTIONS), activation='softmax')
    ])

    dummy_model.compile(optimizer='adam', loss='categoriqcal_crossentropy')

    # Start real-time detection with the dummy model.
    real_time_detection(dummy_model)