import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
    
def cnn():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(96, (11, 11), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(9,activation=tf.keras.activations.softmax))
    return model