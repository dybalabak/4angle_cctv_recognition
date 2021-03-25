import numpy as np
import tensorflow
from data_gen import DataGenerator
from model_ML import create_model_pretrain
from data_helper import readfile_to_dict

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint

###### Parameters setting
dim = (224,224) # for MobileNetV2 or InceptionResnetV2
n_sequence = 8 # for LSTM
n_channels = 3 # color channel(RGB)
n_output = 15 #18 # number of output class
batch_size = 2
n_mul_train = 1 # To increase sample of train set
n_mul_test = 2 # To increase sample of test set

# make sure the the data directory exists
path_dataset = 'data_reduced/'

import os
try:
    os.mkdir(path_dataset)
except:
    pass
######

# Keyword argument
params = {'dim': dim,
          'batch_size': batch_size, # you can increase for faster training
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'option': 'RGBdiff',
          'shuffle': True}


#below  are important parts run program with no interruptions -- 
# set the paths of your train and testlist --> make sure format matches
# train_txt = "dataset_list/trainlist.txt"
# test_txt = "dataset_list/testlist.txt"
train_txt = "dataset_list/general_trainlist.txt"
test_txt = "dataset_list/general_validationlist.txt"

# Read file
# train_d and test_d is dictionary that contain name of video as key and label as value
# For example, {'a01\a01_s08_e01': 0, 'a01\a01_s08_e02': 0, .... }
# It's used for getting label(Y)

#

train_d = readfile_to_dict(train_txt)
test_d = readfile_to_dict(test_txt)

# Prepare key, name of video(X)
train_keys = list(train_d.keys()) * n_mul_train
test_keys = list(test_d.keys()) * n_mul_test

# Generators
training_generator = DataGenerator(train_keys, train_d, **params, type_gen='train')
validation_generator = DataGenerator(test_keys, test_d, **params, type_gen='test')

# Design model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output)
start_epoch = 0

# Load weight of unfinish training model(optional)
load_model = True
if load_model:
    weights_path = 'save_weight/weight-data-20201124-600-0.87-0.86.hdf5' # name of model  -- This trained weight on the kard data datasets
    start_epoch = 600 # currently it should be set at the 0 since there is learned parameter
    model.load_weights(weights_path)

# Set callback
validate_freq = 6
filepath = "save_weight/"+"weight-data-20201124-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, validation_freq=validate_freq, save_freq = "epoch", period = 10)
callbacks_list = [checkpoint]

# # Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=900,
                    callbacks=callbacks_list,
                    initial_epoch=start_epoch,                 
                    validation_steps=int(len(validation_generator)/batch_size)
                    )


