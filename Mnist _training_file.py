#importing important libraries

import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D,Flatten
from tensorflow.keras.utils import to_categorical

#Loading the dataset

(train_img,train_labels),(test_img,test_labels)=mnist.load_data()

#Reshaping the dataset

train_img=train_img.reshape(60000,28,28,1)
test_img=test_img.reshape(10000,28,28,1)
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#Model

model=tf.keras.Sequential([
                        Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
                        MaxPooling2D(2,2),
                        Conv2D(64,(3,3),activation='relu'),
                        MaxPooling2D(2,2),
                        Conv2D(32,(3,3),activation='relu'),
                        MaxPooling2D(2,2),
                        Flatten(),
                        Dense(128,activation='relu'),
                        Dense(64,activation='relu'),
                        Dense(10,activation='softmax') ])

#Compiling

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Training

history=model.fit(train_img,train_labels,epochs=16,validation_data=(test_img,test_labels))

#Plotting the graphs

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid('on')
