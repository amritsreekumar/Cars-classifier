# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob
#from sklearn.model_selection import train_test_split
from tensorflow.python.keras import activations
import pandas as pd
from tensorflow.python.keras.layers.core import Flatten
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, \
Dense, Input, Activation, MaxPool2D, Dropout
from tensorflow.keras import Model
import random
from numpy.random import default_rng
import keras as K



class Datagen(tf.keras.utils.Sequence):
  def __init__(self, list_IDs, label_ids, val_dict, val, batch_size=32, n_classes=200, dim = (64,64), n_channels = 3, shuffle=True):
    self.val = val
    self.dim = dim
    self.label_ids = label_ids
    self.val_dict = val_dict #to map between validation images and their corresponding classes
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.batch_size = batch_size
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()
    self.augmentor = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )


  def __len__(self):
    #number of batches per epoch
    return int(np.floor(len(self.list_IDs)/self.batch_size))

  def __getitem__(self, index):
    ##Select a set of
    indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
    if self.val==False:
      list_IDs_temp = [k for k in indexes]     # Generate data
    else:
      list_IDs_temp = [self.list_IDs[k] for k in indexes]
    #print(list_IDs_temp)
    X, y = self.__data_generation(list_IDs_temp) #to be implemented
    return X, y

  def on_epoch_end(self):
    if self.val == True:
      self.indexes = np.arange(len(self.list_IDs))
    else:
      self.indexes = self.list_IDs
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
        #X[i,] = load_img('tiny-imagenet-200/val/images/' + ID)
        X[i,] = load_img('/home/rahulnai/Workspace/Cars-classifier/tiny-imagenet-200/val/images/' + ID)
        # Store class
        labelid = self.val_dict[ID]
        new_label = self.label_ids[labelid]
        y[i] = new_label
      X = X / 255.
      #X_transformed = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)
      
      #return next(X_transformed), tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
      return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    elif self.val == False: 
      for i, filedata in enumerate(list_IDs_temp):
          # Store sample
          folderName, filename = os.path.basename(filedata).split('.')[0].split('_')
          X[i,] = load_img('/home/rahulnai/Workspace/Cars-classifier/tiny-imagenet-200/train/' + folderName + '/images/' + folderName + '_' + str(filename) +'.JPEG')
          # Store class
          labelid = self.label_ids[folderName]
          y[i] = labelid
      X = X / 255.
      X_transformed = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)
      return next(X_transformed), tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
      #r#eturn X / 255., tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
      # return np.array(X), np.array(y)

  

class_ids = open('/home/rahulnai/Workspace/Cars-classifier/tiny-imagenet-200/wnids.txt', "r")
class_ids = class_ids.readlines()
label_ids = {}
for i in range(len(class_ids)):
  label_ids[class_ids[i].split('\n')[0]] = i

####Full path to all images
train_paths = glob.glob('/home/rahulnai/Workspace/Cars-classifier/tiny-imagenet-200/train/**/*.JPEG', recursive=True)
# train_images = []
# for i in train_paths:
#   ##Extract the filename from the full qualified name
#   ##These are list ids to input generator
#     train_images.append(os.path.basename(i).split('.')[0])
# ##Folder names in the training folder
train_labels = []
for i in train_paths:
  ##Extract the filename from the full qualified name
  ##These are list ids to input generator
    train_labels.append(os.path.basename(i))
    #folderName, filename = os.path.basename(i).split('.')[0].split('_')
    #folderName = os.path.basename(i).split('.')[0].split('_')[0]
    #train_labels[keyp] = folderName
##Folder names in the training folder
# train_labels = []
# for i in train_images:
#     train_labels.append(i.split('_')[0])


val_data = open('/home/rahulnai/Workspace/Cars-classifier/tiny-imagenet-200/val/val_annotations.txt', "r")
val_data = val_data.readlines()
validation_images = []
validation_labels = []
val_dict = {}
for i in range(len(val_data)):
  image_id = val_data[i].split('\t')[0]
  validation_images.append(image_id)
  image_label = val_data[i].split('\t')[1]
  validation_labels.append(image_label)
  val_dict[validation_images[i]] = validation_labels[i]



#X_train, X_test, y_train, y_test = train_test_split(train_labels, train_labels, test_size=0.2, random_state=42, shuffle=True)


# random.shuffle(validation_images)

# X_test = validation_images[:8000]
# X_valid = validation_images[8000:]

X_valid = validation_images[:8000]
X_test = validation_images[8000:]

##pass folder names (train labels), foldername to class mappings(label ids) 
training_generator = Datagen(train_labels, label_ids, val_dict = None, val = False, batch_size=2000)
validation_generator = Datagen(X_valid, label_ids, val_dict, val = True, batch_size=2000)
test_generator = Datagen(X_test, label_ids, val_dict, val = True, batch_size=200)


def model():
  input_img = Input(shape=(64, 64, 3))
  x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), padding='same', activation=None)(input_img)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPool2D(pool_size=(2, 3))(x)
  #x = Dropout(0.5)(x)
  x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPool2D(pool_size=(2, 3))(x)
  #x = Dropout(0.5)(x)
  x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding='same', activation=None)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPool2D(pool_size=(2, 3))(x)
  #x = Dropout(0.5)(x)
  # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation=None)(x)
  # x = BatchNormalization()(x)
  # x = Activation('relu')(x)
  x = Flatten()(x)
  #x = Dense(units=256, activation='relu')(x)
  # output = Dense(units=200, activation='softmax')(x)
  x = Dense(units=1024, activation='sigmoid')(x)
  x = Dropout(0.5)(x)
  x = Dense(units=512, activation='sigmoid')(x)
  # x = Dropout(0.5)(x)
  output = Dense(units=200, activation='softmax')(x)


  return Model(input_img, output)

  

model = model()
print(model.summary())

use_saved_model = True
checkpoint_filepath = 'checkpoint/'
#####################Loading saved model if one exsists
# if not os.path.exists('checkpoint_filepath/saved_model.pb') & use_saved_model:
#     model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"],)
# else:
if use_saved_model:

  # model.load_weights('checkpoint/saved_model.pb') #load the model from file
  model = tf.keras.models.load_model(checkpoint_filepath)
  # print('lr is ', K.get_session().run(model.optimizer.lr))
  loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=0)
  print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))



reduceonplateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.01,
    patience=10,
    verbose=1.0,
    mode="auto",
    min_delta=0.005,
    cooldown=0,
    min_lr=0.0
)

# model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"],)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"],)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs = 100,
                    # initial_epoch=50,
                    callbacks = [model_checkpoint_callback, reduceonplateau] ,
                    workers = 6)
loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=0)
print(loss)
print(acc)