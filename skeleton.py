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
import pandas as pd
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
  def __init__(self, list_IDs, labels, val, batch_size=32, n_classes=200, dim = (64,64), n_channels = 3, shuffle=True):
    self.val = val
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
    if self.val == True:
      for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X[i,] = load_img('tiny-imagenet-200/val/images/' + ID)
        #print(X[i,].shape)
        # Store class
        labelid = val_dict[ID]
        new_label = label_ids[labelid]
          #print(labelid)
        y[i] = self.labels[new_label]
        #print(tf.keras.utils.to_categorical(y, num_classes=self.n_classes))
      return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    elif self.val == False: 
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          X[i,] = load_img('tiny-imagenet-200/train/' + ID + '/images/' + ID + '_' + str(i) +'.jpeg')
          #print(X[i,].shape)
          # Store class
          labelid = label_ids[ID]
          #print(labelid)
          y[i] = self.labels[labelid]
          #print(tf.keras.utils.to_categorical(y, num_classes=self.n_classes))
      return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
      # return np.array(X), np.array(y)

  

class_ids = open('tiny-imagenet-200/wnids.txt', "r")
class_ids = class_ids.readlines()
label_ids = {}
for i in range(len(class_ids)):
  label_ids[class_ids[i].split('\n')[0]] = i

print(label_ids)

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
    newlistlabels.append(label_ids[k])

print(len(newlistlabels))
#print(len(newlistlabels))


val_data = open('tiny-imagenet-200/val/val_annotations.txt', "r")
#val_data.columns = ["image", "class", "x", "y", "w", "h"]
val_data = val_data.readlines()
validation_images = []
validation_labels = []
val_dict = {}
counter2 = 0
for i in range(len(val_data)):
  image_id = val_data[i].split('\t')[0]
  validation_images.append(image_id)

  image_label = val_data[i].split('\t')[1]
  validation_labels.append(image_label)
  val_dict[validation_images[i]] = validation_labels[i]


new_val_list = []
for i in validation_labels:
  new_val_list.append(label_ids[i])


training_generator = Datagen(train_labels, newlistlabels, val = False)
print(newlistlabels[:10])
#new = iter(training_generator)
##print(new)
#change the below to above format train labels
validation_generator = Datagen(validation_images, new_val_list, val = True)
print(validation_images[:10])
def model():
  input_img = Input(shape=(64, 64, 3))
  x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), padding='same', activation=None)(input_img)
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
  output = Dense(units=200, activation='softmax')(x)

  return Model(input_img, output)

model = model()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"])

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs = 10,
                    workers = 6)