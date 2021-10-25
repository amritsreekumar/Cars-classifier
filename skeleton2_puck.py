
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
import Datagen as Datagen


class_ids = open('./tiny-imagenet-200/wnids.txt', "r")
class_ids = class_ids.readlines()
label_ids = {}
for i in range(len(class_ids)):
  label_ids[class_ids[i].split('\n')[0]] = i

####Full path to all images
train_paths = glob.glob('./tiny-imagenet-200/train/**/*.JPEG', recursive=True)
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


val_data = open('./tiny-imagenet-200/val/val_annotations.txt', "r")
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


random.shuffle(validation_images)

X_test = validation_images[:8000]
X_valid = validation_images[8000:]

training_generator = Datagen.Datagen(train_labels, label_ids, val_dict = None, val = False, batch_size=200, flag=1)
validation_generator = Datagen.Datagen(X_valid, label_ids, val_dict, val = True, batch_size=200, flag=0)
test_generator = Datagen.Datagen(X_test, label_ids, val_dict, val = True, batch_size=200, flag=0)


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
  #x = Dropout(0.5)(x)
  output = Dense(units=200, activation='softmax')(x)


  return Model(input_img, output)

  

model = model()
print(model.summary())

use_saved_model = False
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
                    use_multiprocessing=False,
                    epochs = 65,
                    # initial_epoch=50,
                    callbacks = [model_checkpoint_callback],
                    workers = 6)
loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=0)
print(loss)
print(acc)