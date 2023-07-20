import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential, load_model


model_save="./model2.hdf5"
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

def prediction(img):
    img=cv.GaussianBlur(img,(3,3),2)
    img=cv.resize(img, (64,64))
    img=img.astype('float32')/255
    img = np.expand_dims(img, axis=0)
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
            img[center[1]-130:center[1]+130, center[0]-130:center[0]+130]
        return True, img, img_rect
    else:
        return False, img, img
    
def show_frames():
        # Get the latest frame and convert into Image
        global prev_char, word
        cv2image= cv.cvtColor(cap.read()[1],cv.COLOR_BGR2RGB)
        check, img_p, img_r=proces(cv2image)
        if check:
            class_model, predit=prediction(img_p)
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
