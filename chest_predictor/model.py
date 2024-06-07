import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def simple_model():
    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(IMG_HEIGHT,IMG_WIDTH,1)))
    model.add(Conv2D(16, kernel_size=10, activation='relu'))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(32, kernel_size=8, activation="relu"))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(32, kernel_size=6, activation="relu"))
    model.add(MaxPooling2D(3))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
