import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_autoencoder():

    input_img = layers.Input(shape=(128,128,1))

    noisy = layers.GaussianNoise(0.1)(input_img)

    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(noisy)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)

    bottleneck = layers.Dense(128,activation='relu',name="bottleneck")(x)

    x = layers.Dense(16*16*128,activation='relu')(bottleneck)
    x = layers.Reshape((16,16,128))(x)

    x = layers.Conv2DTranspose(128,(3,3),strides=2,padding='same',activation='relu')(x)
    x = layers.Conv2DTranspose(64,(3,3),strides=2,padding='same',activation='relu')(x)
    x = layers.Conv2DTranspose(32,(3,3),strides=2,padding='same',activation='relu')(x)

    decoded = layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder