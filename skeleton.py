# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import os
import glob
from tensorflow.python.keras import activations

from tensorflow.python.keras.layers.core import Flatten


import numpy as np
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

import tensorflow
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, \
Dense, Input, Activation, MaxPool2D
from tensorflow.keras import Model

import sklearn.metrics

from numpy.random import default_rng



class Datagen(tf.keras.utils.Sequence):
  def __init__(self, list_IDs, labels, batch_size=32,n_classes=200, dim = (64,64), n_channels = 3, shuffle=True):
    self.dim = dim
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.batch_size = batch_size
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    #number of batches per epoch
    return int(np.floor(len(self.list_IDs)/self.batch_size))

  def __getitem__(self, index):
    indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size] 
    list_IDs_temp = [self.list_IDs[k] for k in indexes]     # Generate data
    #print(list_IDs_temp)
    X, y = self.__data_generation(list_IDs_temp) #to be implemented
    return X, y

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  #to load the data from the indices we give it
  #indices should be given in the main script
  def __data_generation(self, list_IDs_temp):
    X = np.zeros((self.batch_size, *self.dim, self.n_channels))
    y = np.zeros((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X[i,] = load_img('tiny-imagenet-200/train/' + ID + '/images/' + ID + '_' + str(i) +'.jpeg')
        #print(X[i,].shape)
        # Store class
        labelid = newdictlabels[ID]
        #print(labelid)
        y[i] = self.labels[labelid]
    return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    # return np.array(X), np.array(y)

  

train_paths = glob.glob('tiny-imagenet-200/train/**/*.JPEG', recursive=True)
train_images = []
for i in train_paths:
    train_images.append(os.path.basename(i).split('.')[0])

train_labels = []
newdictlabels = {}
newlistlabels = []
counter = 0
for i in train_images:
    train_labels.append(i.split('_')[0])
    k = i.split('_')[0]
    if k not in newdictlabels:
      newdictlabels[k] = counter
      counter = counter + 1
    newlistlabels.append(newdictlabels[k])

#print(len(newlistlabels))



training_generator = Datagen(train_labels, newlistlabels)
#new = iter(training_generator)
##print(new)
validation_generator = Datagen(train_labels[:50], newlistlabels[:50])


#print(training_generator)


# model = Sequential([
#   tf.keras.Input(shape = (64,64,3)),
#   layers.Conv2D(32,kernel_size=(3,3),activation = "relu"),
#   layers.MaxPooling2D(pool_size=(2,2)),
#   layers.Flatten(),
#   layers.Dense(200,activation = "softmax")
# ])

def get_keras_model():
  input_img = Input(shape=(64, 64, 3))
  x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), padding='same', activation=None)(input_img)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPool2D(pool_size=(4, 4))(x)
  x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  output = Dense(units=10, activation='softmax')(x)

  return Model(input_img, output)

model = get_keras_model()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"])

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    )