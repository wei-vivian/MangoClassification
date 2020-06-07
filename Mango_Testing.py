#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:24:25 2020

@author: weilai
"""

# import the necessary packages
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array


import numpy as np
import cv2
import csv


def build_model(IMG_WIDTH, IMG_HEIGHT, Depth, classes, finalAct="softmax"):
    
    model = Sequential()
    inputShape = (IMG_HEIGHT, IMG_WIDTH, Depth)
    chanDim = -1
	
    if K.image_data_format() == "channels_first":
        inputShape = (Depth, IMG_HEIGHT, IMG_WIDTH)
        chanDim = 1
    
    # CONV(32) => RELU => POOL
    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
        
    # (CONV(64) => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
        
	# (CONV(128) => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation(finalAct))
	
    return model
#==============================================================================
if __name__ == '__main__':
    
    PATH = './Mango_Class2/C1-P1_Dev'
    count = 0
    test_list = []
    batch_size = 32
    epochs = 80
    IMG_DIMS = (96, 96, 3)
    INIT_LR = 0.001
    
    # Read testing data
    with open('./Mango_Class2/dev.csv', encoding='utf-8', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            count+=1
            test_list.append(row)
            
    """
    model = load_model('Mango_v4.h5')
    """
    
    Class = ['A', 'B', 'C']
    
    # Build model
    model = build_model( IMG_WIDTH = IMG_DIMS[1], 
                         IMG_HEIGHT = IMG_DIMS[0],
                         Depth = IMG_DIMS[2], 
                         classes = len(Class),
                         finalAct = 'softmax' )
    # Load model weights
    model.load_weights("weights.best3.hdf5")
    
    correct = 0
    confusion_matrix = np.zeros((3, 3), dtype = int)
    
    # Prediction for each testing data
    for i in range(1, len(test_list)):
   
        #print(PATH+'/'+str(test_list[i][0]))
        image = cv2.imread(PATH+'/'+str(test_list[i][0]))
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        print("[INFO] Classifying image...")
        proba = model.predict(image)
        print(proba)
        print(test_list[i])
        print(Class[proba.argmax()])
        
        if Class[proba.argmax()] == test_list[i][1]:
            correct += 1
        j = 0
        if test_list[i][1] == 'B':
            j = 1
        elif test_list[i][1] == 'C':
            j = 2
        confusion_matrix[j][proba.argmax()] += 1
    
    # Evaluation
    print(correct)
    print(confusion_matrix)
   
    