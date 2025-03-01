import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# ======================
# Configuration
# ======================
# List of actions for recognition
ACTIONS = ['I', 'Hungry', 'Want', 'Food', 'Go', 'House', 'Sit', 'You', 'Right', 'Need',
           'Water', 'Long', 'Time', 'Person', 'Old_age', 'Old_time', 'Now', 'Here', 'Walk',
           'Swim', 'My', 'Catch', 'Fire', 'Zoo', 'Morning', 'Father']
DATA_PATH = '../temp/training_data'
SEQ_LENGTH = 30
NUM_SEQUENCES = 30

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ======================
# Common Functions
# ======================
def mediapipe_detection(image, model):
    """
    Process image with MediaPipe Holistic model.
    Converts image to RGB, processes it, and then returns a BGR image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    """
    Extract and flatten keypoints from MediaPipe results.
    Returns concatenated arrays for pose, face, and both hands.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def create_model(input_shape, num_classes):
    """
    Build an LSTM model using the provided input shape and
    number of output classes.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(256, return_sequences=True),
        LSTM(128),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model