
import os
from cv2 import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation
from keras.callbacks import ModelCheckpoint

data_path = "./dataset"
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

img_size = 100
data = []
target = []

for cat in categories:
    folder_path = os.path.join(data_path, cat)
    img_names = os.listdir(folder_path)

    for name in img_names:
        img_path = os.path.join(folder_path, name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[cat])
        except Exception as e:
            print("Exception: ", e)

data = np.array(data)/255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

from keras.utils import np_utils

new_target = np_utils.to_categorical(target)

np.save('data', data)
np.save('target', new_target)

data_after = np.load('data.npy')
target_after = np.load('target.npy')

model = Sequential()
model.add(Conv2D(200, (3,3), input_shape=data_after.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(100, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

train_data, test_data, train_target, test_target = train_test_split(data_after, target_after, test_size=0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor="val_loss", verbose=0, save_best_only=True, mode="auto")
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint], validation_split=0.2)