import cv2
import numpy as np
import tensorflow as tf
from model_utils import (ACTIONS, SEQ_LENGTH, mp_holistic,
                         mediapipe_detection, extract_keypoints)

def real_time_detection(model):
    """
    Run real-time sign detection using a loaded model.
    Captures video frames, extracts keypoints, then predicts the sign.
    """
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8  # Confidence threshold for predictions

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == SEQ_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if res[np.argmax(res)] > threshold:
                    current_action = ACTIONS[np.argmax(res)]
                    if not sentence or (sentence[-1] != current_action and
                                        np.unique(predictions[-10:]).size == 1):
                        sentence.append(current_action)
                        sentence = sentence[-3:]

                sequence = sequence[-(SEQ_LENGTH-1):]

            # Display the prediction on the image
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the pre-trained model
    model = tf.keras.models.load_model('model.h5')
    real_time_detection(model)