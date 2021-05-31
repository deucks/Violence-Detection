import os
import scipy
import tensorflow as tf
import zipfile

import random
import keras
import glob
from keras_video import OpticalFlowGenerator, VideoFrameGenerator, SlidingFrameGenerator 
import keras_video.utils

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
from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D
from collections import deque
import sys
import logging
from keras.applications import Xception, ResNet50, InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import TimeDistributed, GRU, Dense, Dropout

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

from keras.applications.vgg16 import VGG16


os.environ['PATH']


# dir_images = "C:\\Users\\raaji\\Desktop\\ViolenceDetection\\dataset_new"
# dir_train = os.path.join(dir_images, 'Train')
# dir_test = os.path.join(dir_images, 'Test')

# num_classes = len(os.listdir(dir_train))
# print(f"Number of classes is {num_classes}")

# use sub directories names as classes
#classes = [i.split(os.path.sep)[1] for i in glob.glob('dataset/*')]
classes = [i.split(os.path.sep)[1] for i in glob.glob('E:/archive/rlvd/*')]
classes.sort()
# some global params
SIZE = (112, 112)
CHANNELS = 3
NBFRAME = 5
BS = 8
# pattern to get videos and classes
glob_pattern='E:/archive/rlvd/{classname}/*.mp4'
#glob_pattern='dataset/{classname}/*.avi'
# for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.2,
    horizontal_flip=True,
    rotation_range=5,
    width_shift_range=.2,
    height_shift_range=.2)
# # Create video frame generator

train = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split=.30, 
    shuffle=True,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=True)

# train = OpticalFlowGenerator(
#     classes=classes, 
#     flowlevel=3,
#     winsize=15,
#     iterations = 3,
#     glob_pattern=glob_pattern,
#     nb_frames=NBFRAME,
#     split=.50, 
#     shuffle=True,
#     batch_size=BS,
#     target_shape=SIZE,
#     nb_channel=CHANNELS,
#     #transformation=data_aug,
#     use_frame_cache=True)

# train = VideoFrameGenerator(
#     classes=classes, 
#     glob_pattern=glob_pattern,
#     nb_frames=NBFRAME,
#     split=.45, 
#     shuffle=True,
#     batch_size=BS,
#     target_shape=SIZE,
#     nb_channel=CHANNELS,
#     transformation=data_aug,
#     use_frame_cache=True)
#valid = train.get_validation_generator()
valid = train

#keras_video.utils.show_sample(train)


# train_datagen = ImageDataGenerator(
# rescale=1./255,
#       rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.4
# )

# ## Start Your Code Here ###
# dir_images = "Fruit-Images-Dataset-master"
# dir_train = os.path.join(dir_images, 'Training')
# dir_test = os.path.join(dir_images, 'Test')
# # Flow training images in batches of 20 using train_datagen generator
# train_generator = train_datagen.flow_from_directory(
#         dir_train,  # This is the source directory for training images
#         target_size=(224,224),  # Complete the code, All images will be resized to 224x224
#         batch_size=32,
#         # Since we use binary_crossentropy loss, we need binary labels
#         class_mode='categorical')

# # Flow validation images in batches of 20 using test_datagen generator
# validation_generator = test_datagen.flow_from_directory(
#         dir_test,
#         target_size=(224,224), # Complete the code, All images will be resized to 224x224
#         batch_size=32,
#         class_mode='categorical')

def build_mobilenet(shape=(224, 224, 3), nbout=3):
    model = keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=shape,
        weights='imagenet')
    # Keep 9 layers to train﻿﻿
    trainable = 9
    for layer in model.layers[:-trainable]:
        layer.trainable = False
    for layer in model.layers[-trainable:]:
        layer.trainable = True
    output = GlobalMaxPool2D()
    return keras.Sequential([model, output])

def build_convnet(shape=(112, 112, 3)):
    momentum = .9
    model = keras.Sequential()

    model.add(Conv2D(64, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())
    
    model.add(Conv2D(1024, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(1024, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # model.add(MaxPool2D())
    
    # model.add(Conv2D(2048, (3,3), padding='same', activation='relu'))
    # model.add(Conv2D(2048, (3,3), padding='same', activation='relu'))
    # model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model

def action_model(shape=(5, 112, 112, 3), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = ResNet50(include_top=False, weights='imagenet', pooling='max')
    #convnet = build_convnet(shape[1:])
    #convnet = build_mobilenet(shape[1:])
    # vgg = VGG16(
    #     include_top=False,
    #     weights='imagenet',
    #     input_shape=(112, 112, 3)
    # )
    # for layer in vgg.layers[:-4]:
    #     layer.trainable = False
    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    # model.add(TimeDistributed(convnet, input_shape=shape))
    model.add(TimeDistributed(convnet, input_shape=shape))
    # model.add(
    #     TimeDistributed(
    #         Flatten()
    #     )
    # )
    # here, you can also use GRU or LSTM
    # model.add(LSTM(256))
    model.add(LSTM(2, activation='relu', return_sequences=False))
    # and finally, we make a decision network
    #model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(.1))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(.2))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(.2))
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(.2))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(nbout, activation='sigmoid'))

    print(model.summary())
    #model.add(Dense(nbout, activation='softmax'))
    return model

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
model = action_model(INSHAPE, len(classes))
model.summary()
optimizer = keras.optimizers.Adam(0.001)
model.compile(
    optimizer,
    'binary_crossentropy',
    metrics=['acc']
)


# model.compile(
#     optimizer,
#     'categorical_crossentropy',
#     metrics=['acc']
# )

EPOCHS=200
# create a "chkp" directory before to run that
# because ModelCheckpoint will write models inside
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    # keras.callbacks.ModelCheckpoint(
    #     'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    #     verbose=1),
]
model.fit_generator(
    train,
    validation_data=valid,
    verbose=1,
    steps_per_epoch=40, 
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save('model.h5')