import os
import scipy
import tensorflow as tf
import zipfile

import random


from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf

from keras import Input
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys
import logging
from keras.applications import Xception, ResNet50, InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras_video import VideoFrameGenerator




os.environ['PATH']


dir_images = "C:\\Users\\raaji\\Desktop\\ViolenceDetection\\data"
dir_train = os.path.join(dir_images, 'Train')
dir_test = os.path.join(dir_images, 'Test')

num_classes = len(os.listdir(dir_train))
print(f"Number of classes is {num_classes}")


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)

## Start Your Code Here ###

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        dir_train,  # This is the source directory for training images
        target_size=(224,224),  # Complete the code, All images will be resized to 224x224
        batch_size=32,
        color_mode="rgb",
        seed=42,
        shuffle=False,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        dir_test,
        target_size=(224,224), # Complete the code, All images will be resized to 224x224
        batch_size=32,
        color_mode="rgb",
        seed=42,
        shuffle=False,
        class_mode='categorical')

inception_base = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# # add a global spatial average pooling layer
# x = inception_base.output
# x = GlobalAveragePooling2D()(x)
# # add a fully-connected layer
# x = Dense(512, activation='relu')(x)
# # and a fully connected output/classification layer
# predictions = Dense(5, activation='softmax')(x)   
# create the full network so we can train on it
#cnn = cnn_class(weights='imagenet', include_top=False,input_shape =(224, 224, 3))
#cnn = TimeDistributed(cnn)(input_layer)
#the resnet output shape is 1,1,20148 and need to be reshape for the ConvLSTM filters
# if cnn_class.__name__ == "ResNet50":
    # cnn = Reshape((seq_len,4, 4, 128), input_shape=(seq_len,1, 1, 2048))(cnn)'


weights = 'imagenet'

dropout = 0.0

lstm_setup = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False))

# #x = inception_base.output

# input_layer = Input(shape=(2, 224, 224, 3))

# cnn = TimeDistributed(lstm_setup)(input_layer)

# lstm = lstm_setup[0](kernel_size=(3, 3),filters=256,padding='same',return_sequences=False)(cnn)

# maxpool = MaxPooling2D(pool_size=(2, 2))(lstm)
# flat = Flatten()(lstm)
baseModel = applications.VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in baseModel.layers:
    layer.trainable = False
x = baseModel.output

# LSTM layer

#print(Shape(x))
#x = Conv2D(kernel_size=(3, 3), filters=16)(x)
#x = Flatten(name="flatten")(x)

x = Reshape((49, 512))(x)

#x = ((ConvLSTM2D(512, kernel_size=(3, 3), activation="relu", return_sequences=True, trainable=False)))(x)
x = ((LSTM(512, activation="relu", return_sequences=True, trainable=False)))(x)



#     x = Dropout(0.5)(x)

# x = Dropout(dropout)(x)
# x = Dense(1000)(x)
# x = Activation('relu')(x)
# x = Dense(256)(x)
# x = Dropout(dropout)(x)
# x = Activation('relu')(x)
# x = Dense(10)(x)
# x = Dropout(dropout)(x)
# x = Activation('relu')(x)
# add a fully-connected layer
#x = Dense(512, activation='relu')(x)



# FC layer
x = Flatten(name="flatten")(x)

# fc1 layer
x = Dense(units=1000, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# fc2 layer
x = Dense(units=256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# fc2 layer
x = Dense(units=10, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# Output layer
output = Dense(units=5, activation='softmax')(x)

model = Model(inputs=baseModel.input, outputs=output) 
opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06, clipnorm=1.)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=200,  # 2000 images = batch_size * steps
    epochs=20,
    validation_data=validation_generator,
    validation_steps=100, 
    shuffle = False,# 1000 images = batch_size * steps
    #callbacks = [checkpoint],
    verbose=1
)


# # add a global spatial average pooling layer
# x = inception_base.output

# #x = lstm_setup[0](kernel_size=(3, 3),filters=256,padding='same',return_sequences=False, input_shape=(None, 2, 224, 224, 3))(x)
# #x = Input(shape=(None, 244,244,3))
# x = TimeDistributed(ResNet50(weights=weights, include_top=False,input_shape =(224, 224, 3)))(x)
# x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True)(x)

# x = MaxPooling2D(pool_size=(2,2))(x)
# #x = GlobalAveragePooling2D()(x)
# #x = Flatten()(x)

# x = BatchNormalization()(x)
# x = Dropout(dropout)(x)
# x = Dense(1000)(x)


# x = Activation('relu')(x)
# x = Dense(256)(x)
# x = Dropout(dropout)(x)
# x = Activation('relu')(x)
# x = Dense(10)(x)
# x = Dropout(dropout)(x)
# x = Activation('relu')(x)
# # add a fully-connected layer
# #x = Dense(512, activation='relu')(x)
# # and a fully connected output/classification layer
# predictions = Dense(5, activation='sigmoid')(x)   
# create the full network so we can train on it

# inception_transfer = Model(inputs=inception_base.input, outputs=predictions)

# inception_transfer.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])

# history = inception_transfer.fit_generator(
#     train_generator,
#     steps_per_epoch=200,  # 2000 images = batch_size * steps
#     epochs=20,
#     validation_data=validation_generator,
#     validation_steps=32, 
#     shuffle = True,# 1000 images = batch_size * steps
#     #callbacks = [checkpoint],
#     verbose=1
# )