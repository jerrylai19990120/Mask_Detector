import tensorflow as tf
import pathlib
import numpy as np
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from cv2 import cv2
from keras.models import load_model

model = load_model('model-009.model')
labels_dict = {0:"MASK", 1:"NO MASK"}
colors_dict = {0: (0,255,0), 1:(0,0,255)}

face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(0)

while True:
    ret, frame = videoCapture.read()

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (face_x, face_y, w, h) in face_rects:
        face_img = gray[face_y:face_y+h, face_x:face_x+w]
        resized = cv2.resize(face_img, (100,100))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)
            
        label = np.argmax(result, axis=1)[0]
            
        cv2.rectangle(frame, (face_x, face_y), (face_x+w,face_y+h), (0,0,255), 5)
        cv2.putText(frame, labels_dict[label], (face_x, face_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,colors_dict[label],2)
    

    cv2.imshow('mask detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()





