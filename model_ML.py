from tensorflow import keras
import tensorflow as tf
#tf.compat.v1 =
from keras import Sequential
from tensorflow.keras.layers import LSTM
from keras.layers import Dense, TimeDistributed
from keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.python.keras import optimizers
from keras.applications import MobileNetV2, InceptionResNetV2, InceptionV3, ResNet50
from keras import backend as K

from keras.models import Model

# get the model inorder to train or load weights to evaluate and use the model.
def create_model_pretrain(dim, n_sequence, n_channels, n_output):
    '''
    Paremeter:
        dim -- dimension of image, use (224,224) for MobileNetV2
        n_sequence -- number of sequence(timestep) for LSTM
        n_channels -- number of color channels, use 3 for RGB
        n_output -- number of output class
    '''
    model = Sequential()
    print("*** n_sequence, *dim, n_channels: ")
    print((n_sequence, *dim, n_channels))
    model.add( 
        TimeDistributed(
            InceptionResNetV2(weights='imagenet',include_top=False), 
            input_shape=(n_sequence, *dim, n_channels)
        )
    )
    model.add(
        TimeDistributed(
            GlobalAveragePooling2D()
        )
    )
    print("*** model summary: ")
    print(model.summary())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(.5))  
    model.add(Dense(n_output, activation='softmax'))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

