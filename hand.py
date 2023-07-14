import mediapipe as mp
import cv2 as cv
import numpy as np
from tensorflow.keras.models import Sequential, load_model

model_save="./model.hdf5"
classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']

model=load_model(model_save)

def prediction(img):
    img=cv.resize(img, (48,48))
    img=img.astype('float32')/255
    pred=model.predict(img)
    return classes[np.argmax(pred)], pred