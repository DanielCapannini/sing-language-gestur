import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential, load_model
import csv
import copy
import argparse
import itertools
import os
from collections import Counter
from collections import deque



model_save="./model10.hdf5"
classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']

model=load_model(model_save)
prev_char=''
word=""

cap= cv.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
webcam=cv.VideoCapture(0)
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.5,
    )

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def prediction(img):

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
            img_rect = cv.rectangle(img, (center[0] - 120, center[1] - 120), (center[0] + 120, center[1] + 120), (0, 255, 0), 2)
            debug_image = copy.deepcopy(img)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_landmark_list = np.expand_dims(pre_processed_landmark_list, axis=0)
            pred=model.predict(pre_processed_landmark_list)
        return True, img_rect, pred
    else:
        return False, img, img
    
def show_frames():
        # Get the latest frame and convert into Image
        global prev_char, word
        cv2image= cv.cvtColor(cap.read()[1],cv.COLOR_BGR2RGB)
        check, img_r, predit=proces(cv2image)
        class_model=classes[np.argmax(predit)]
        if check:
            if np.max(predit)>0.6:
                print(class_model)
                print(np.max(predit))
                p.delete("1.0", "end")
                p.insert('end', np.max(predit))
                if prev_char !=class_model:
                    if class_model == "space":
                        word += ' '
                        t.delete("1.0", "end")
                        t.insert('end', word)
                    elif class_model == "del":
                        word[0:-2]
                        t.delete("1.0", "end")
                        t.insert('end', word)
                    else:
                        word += class_model
                        t.delete("1.0", "end")
                        t.insert('end', word)
                l.delete("1.0", "end")
                l.insert('end', class_model)
                prev_char=class_model
        img = Image.fromarray(img_r)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image = img, master=root)
        GLabel_963.imgtk = imgtk
        GLabel_963.configure(image=imgtk)
        # Repeat after an interval to capture continiously
        GLabel_963.after(20, show_frames)
    


root = tk.Tk()
#setting title
root.title("undefined")
 #setting window size
width=1000
height=500
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
root.geometry(alignstr)
root.resizable(width=False, height=False)

GLabel_963=tk.Label(root)
ft = tkFont.Font(family='Times',size=10)
GLabel_963["font"] = ft
GLabel_963["fg"] = "#333333"
GLabel_963["justify"] = "center"
GLabel_963["text"] = "label"
GLabel_963.place(x=0,y=0,width=500,height=500)

GMessage_100=tk.Message(root, text="precizione:")
GMessage_100.place(x=550,y=50,width=100,height=30)

p = tk.Text(root)
p.config(font =('Times', 10))
p.place(x=700,y=50,width=125,height=30)

GMessage_870=tk.Message(root, text="lettura:")
GMessage_870.place(x=550,y=130,width=100,height=30)

l = tk.Text(root)
l.config(font =('Times', 10))
l.place(x=700,y=130,width=125,height=30)


GMessage_50=tk.Message(root, text="testo:")
GMessage_50.place(x=550,y=210,width=100,height=30)

t = tk.Text(root)
t.config(font =('Times', 10))
t.place(x=700,y=210,width=200,height=150)



show_frames()
root.mainloop()
