#useful library imports

import cv2
import numpy as np
from model_ML import create_model_pretrain
import time
from data_helper import calculateRGBdiff
import pandas as pd
data = []
import tensorflow as tf

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

###### Parameters setting
dim = (224,224) # MobileNetV2 or InceptionResnetV2
n_sequence = 8
n_channels = 3 # RGB Channel
n_output = 15 # number of output class
weights_path = 'save_weight/weight-data-20201124-720-0.91-0.87.hdf5' # pretrain weight path, best weight
#######

def get_class_map():
    classes = []
    indexes = []
    index = 1
    with open("dataset_list/general_validationlist.txt") as file:
        for line in file.readlines():
            #print(line)
            try:
                #print(line.split("/")[1].split("_")[0])
                if line.split("/")[0] not in classes and line.split("/")[0]!= "\n":
                    classes.append(line.split("/")[0])
                    indexes.append(index)
                    index+=1
            except:
                pass
    index = 0
    for cl in classes:
        classes[index] = str(indexes[index])+ " "+cl
        index+=1
    return classes
### load model
model = None
with tf.device("GPU:0"):
    # GPU name can be different on system to system
    # GPU assingment, Since this is computationally very expensive task, only overcome able with 
    # usage of gpu for the prediction for real time test
    # THIS IS THE LATEST AND BEST THAT WAS WORKING ON Kaggle with GPU 0
    # only two assingments are availble there ==? 
    # GPU:0 -- assigned for the task -->  if or --> XLA_GPU:0  should be assigned -- . for the task .
    
    # model, this should exactly match the model generated from the model_ML.py
    
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
    model.load_weights(weights_path)

    ### Define empty sliding window
    frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel

    ### State Machine Define
    RUN_STATE = 0
    WAIT_STATE = 1
    SET_NEW_ACTION_STATE = 2
    state = RUN_STATE # 
    previous_action = -1 # no action
    text_show = 'no action'

    ### Class label define
    class_text = [
    '1 Horizontal arm wave',
    '2 High arm wave',
    '3 Two hand wave',
    '4 Catch Cap',
    '5 High throw',
    '6 Draw X',
    '7 Draw Tick',
    '8 Toss Paper',
    '9 Forward Kick',
    '10 Side Kick',
    '11 Take Umbrella',
    '12 Bend',
    '13 Hand Clap',
    '14 Walk',
    '15 Phone Call',
    '16 Drink',
    '17 Sit down',
    '18 Stand up']
    
    
    # second definition based on the data generated on the system.
    class_text = get_class_map()
    print(class_text)
    # starting to capture video or real time, if realtime put 0
    
    cap = cv2.VideoCapture("action2/drink/cam_1.mp4")# capture from the videos --> 
    start_time = time.time()
    while(cap.isOpened()):
        total = time.time()
        ret, frame = cap.read()  
        
        if ret == True:
            start = time.time()
            new_f = cv2.resize(frame, dim)
            #print(time.time()-start, "time taken for resizing")
            resizing = time.time()-start
            #start = time.time()
            new_f = new_f/255.0
            new_f_rs = np.reshape(new_f, (1, *new_f.shape))
            frame_window = np.append(frame_window, new_f_rs, axis=0)
            

            ### if sliding window is full(8 frames), start action recognition
            if frame_window.shape[0] >= n_sequence:
                frame_window_dif = calculateRGBdiff(frame_window.copy())
                frame_window_new = frame_window_dif.reshape(1, *frame_window_dif.shape)
                # print(frame_window_new.dtype)
                ### Predict action from model
                start_time_predict = time.time()
                #output = None
                #with tf.device("/:XLA_GPU:0"):
                output = model.predict(frame_window_new)[0]     
                #print("Each Prediction took ", time.time()-start_time_predict)
                #start_time_predict = time.time()
                pred_time = time.time()-start_time_predict
                predict_ind = np.argmax(output)
                from_prediction = time.time()
                ### Check noise of action
                if output[predict_ind] < 0.55:
                    new_action = -1 # no action(noise)
                else:
                    new_action = predict_ind # action detect

                ### Use State Machine to delete noise between action(just for stability)
                ### RUN_STATE: normal state, change to wait state when action is changed
                if state == RUN_STATE:
                    if new_action != previous_action: # action change
                        state = WAIT_STATE
                        start_time = time.time()     
                    else:
                        if previous_action == -1:
                            text_show = 'no action'                                              
                        else:
                            text_show = "{: <22}  {:.2f} ".format(class_text[previous_action],
                                        output[previous_action] )
                        print(text_show)  

                ### WAIT_STATE: wait 0.5 second when action from prediction is change to fillout noise
                elif state == WAIT_STATE:
                    dif_time = time.time() - start_time
                    if dif_time > 0.5: # wait 0.5 second
                        state = RUN_STATE
                        previous_action = new_action

                ### put text to image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text_show, (30,100), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)   
                
                ### shift sliding window
                frame_window = frame_window[1:n_sequence]
                
                ### To show dif RGB image
                # vis = np.concatenate((new_f, frame_window_new[0,n_sequence-1]), axis=0)
                # cv2.imshow('Frame', vis)
                cv2.imshow('Frame', frame)
                #print("Time from prediction : ", time.time() - from_prediciton)
                after_prediction = time.time()- from_prediction
                total = time.time()-total
                data.append([resizing, pred_time, after_prediction, total])
                

            ### To show FPS
            # end_time = time.time()
            # diff_time =end_time - start_time
            # print("FPS:",1/diff_time)
            # start_time = end_time
            
            
            
            
            """
            make sure below controller uncommented on realtime test on keyboard controlled machine.
            
            """
     
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break 
        else: 
            break

#system performance for the evaluation for any interruptions.
df = pd.DataFrame(data, columns = ["resizing", "prediction", "from_prediction", "total"])
df.to_csv("performance.csv", index = False)
 
cap.release()

