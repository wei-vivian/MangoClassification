#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:50:54 2020

@author: weilai
"""
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
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

from sklearn.preprocessing import MultiLabelBinarizer

def read_split_data(PATH, img_list, IMAGE_DIMS, class_amount):
    print("[INFO] loading spliting images...")
    # initialize the data and labels
   
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    
    a = 0
    b = 0 
    c = 0
  
    for i in range(1, len(img_list)):
        
        image = cv2.imread(PATH + '/' + str(img_list[i][0]))
        print(PATH + '/' +str(img_list[i][0]))
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        if img_list[i][1] == 'A' and a < (class_amount[0] * 0.8):
            train_data.append(image)
            train_labels.append(img_list[i][1])
            a += 1
        elif img_list[i][1] == 'A' and a >= (class_amount[0] * 0.8):
            dev_data.append(image)
            dev_labels.append(img_list[i][1])
            a += 1
        if img_list[i][1] == 'B' and b < (class_amount[1] * 0.8):
            train_data.append(image)
            train_labels.append(img_list[i][1])
            b += 1
        elif img_list[i][1] == 'B' and b >= (class_amount[1] * 0.8):
            dev_data.append(image)
            dev_labels.append(img_list[i][1])
            b += 1
        if img_list[i][1] == 'C' and c < (class_amount[2] * 0.8):
            train_data.append(image)
            train_labels.append(img_list[i][1])
            c += 1
        elif img_list[i][1] == 'C' and c >= (class_amount[2] * 0.8):
            dev_data.append(image)
            dev_labels.append(img_list[i][1])
            c += 1
            
    # scale training data's pixel intensities to the range [0, 1]
    train_data = np.array(train_data, dtype="float") / 255.0
    train_labels = np.array(train_labels)
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_labels)
    print(train_labels)
        
    # scale dev data's pixel intensities to the range [0, 1]
    dev_data = np.array(dev_data, dtype="float") / 255.0
    dev_labels = np.array(dev_labels)
    mlb = MultiLabelBinarizer()
    dev_labels = mlb.fit_transform(dev_labels)
    
    return train_data, dev_data, train_labels, dev_labels, mlb
    

def build_model(IMG_WIDTH, IMG_HEIGHT, Depth, classes, finalAct="softmax"):
    
    model = Sequential()
    inputShape = (IMG_HEIGHT, IMG_WIDTH, Depth)
    chanDim = -1
	
    if K.image_data_format() == "channels_first":
        inputShape = (Depth, IMG_HEIGHT, IMG_WIDTH)
        chanDim = 1
   
    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

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
    
    batch_size = 32
    epochs = 150
    IMG_DIMS = (96, 96, 3)
    INIT_LR = 0.001
    PATH = './Mango_Class2/C1-P1_Train'
    
    img_list = []
    count = 0
    classA = 0
    classB = 0
    classC = 0
    class_amount = [0, 0, 0]
 
    # Read raw data
    with open('./Mango_Class2/train.csv', encoding='utf-8', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            count+=1
            img_list.append(row)
            if row[1] == 'A':
                class_amount[0] += 1
            elif row[1] == 'B':
                class_amount[1] += 1
            elif row[1] == 'C':
                class_amount[2] += 1
    print(img_list[0][0])
    print(class_amount)
    
    # Split data into training and validation data
    trainX, testX, trainY, testY, mlb = read_split_data( PATH, 
                                                         img_list, 
                                                         IMG_DIMS, 
                                                         class_amount )
   
    # Construct the image generator for data augmentation
    aug = ImageDataGenerator( rotation_range=25, 
                              width_shift_range=0.1,
                              height_shift_range=0.1, 
                              shear_range=0.2, 
                              zoom_range=0.2,
                              horizontal_flip=True, 
                              fill_mode="nearest" )
    
    print("[INFO] Compiling model...")
    model = build_model( IMG_WIDTH = IMG_DIMS[1], 
                         IMG_HEIGHT = IMG_DIMS[0],
                         Depth = IMG_DIMS[2], 
                         classes = len(mlb.classes_),
                         finalAct = 'softmax' )
    
    # Initialize the optimizer
    opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)

    model.compile( loss='binary_crossentropy', 
                   optimizer=opt,
                   metrics=["accuracy"] )
    
    # Checkpoint: Save the best weight condition while training
    filepath="weights.best3.hdf5"
    checkpoint = ModelCheckpoint( filepath, 
                                  monitor='val_accuracy', 
                                  verbose=1, 
                                  save_best_only=True,
                                  mode='max' )
    callbacks_list = [checkpoint]
    
    # Train the network
    print("[INFO] Training network...")
    history = model.fit_generator( aug.flow(trainX, trainY, batch_size = batch_size),
                                   validation_data = (testX, testY),
                                   steps_per_epoch = len(trainX) // batch_size,
                                   epochs = epochs, 
                                   verbose = 1,
                                   callbacks = callbacks_list)
    
    # Save the model
    print("[INFO] Serializing network...")
    model.save('Mango_v3.h5')
   
    # Plot the training loss and accuracy
    print(history.history.keys())
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()

    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
    
   