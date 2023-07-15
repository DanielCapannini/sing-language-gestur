import mediapipe as mp
import cv2 as cv
import numpy as np
from tensorflow.keras.models import Sequential, load_model

model_save="./model.hdf5"
classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']

model=load_model(model_save)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
webcam=cv.VideoCapture(0)
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.5,
    )

def prediction(img):
    img=cv.resize(img, (48,48))
    img=img.astype('float32')/255
    pred=model.predict(img)
    return classes[np.argmax(pred)], pred

def proces(img):
    results=hands.process(img)
    img_height, img_width, _ = img.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]
            center = np.array([np.mean(x) * img_width, np.mean(y) * img_height]).astype('int32')
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_rect = cv.rectangle(img, (center[0] - 120, center[1] - 120), (center[0] + 120, center[1] + 120), (0, 255, 0), 2)
            img[center[1]-130:center[1]+130, center[0]-130:center[0]+130]
            img=cv.resize(img, (48,48))
        return True, img, img_rect
    else:
        return False, img, img
    


while True:
    ret, frame=webcam.read()
    if not ret:
        break

    