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
# from sklearn.model_selection import train_test_split
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
# import resnetkeras
# from resnet import ResNet18
# from resCubs1 import ResNet18Cubs1
# from cubs1cubs2resnet import cubs1cubs2res1
from cubsArch1 import cubsArch1
import DatagenAugndUnAug



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
    # folderName, filename = os.path.basename(i).split('.')[0].split('_')
    # folderName = os.path.basename(i).split('.')[0].split('_')[0]
    # train_labels[keyp] = folderName
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

# X_train, X_test, y_train, y_test = train_test_split(train_labels, train_labels, test_size=0.2, random_state=42, shuffle=True)


# random.shuffle(validation_images)

# X_test = validation_images[:8000]
# X_valid = validation_images[8000:]

X_valid = validation_images[:8000]
X_test = validation_images[8000:]

##pass folder names (train labels), foldername to class mappings(label ids)
training_generator = DatagenAugndUnAug.Datagen(train_labels, label_ids, val_dict=None, val=False, batch_size=1000, flag=1)
validation_generator = DatagenAugndUnAug.Datagen(X_valid, label_ids, val_dict, val=True, batch_size=1000, flag=0)
test_generator = DatagenAugndUnAug.Datagen(X_test, label_ids, val_dict, val=True, batch_size=200, flag=0)

model = cubsArch1(200)

model.build(input_shape=(100, 64, 64, 3))

print(model.summary())

use_saved_model = False
# checkpoint_filepath = 'resnet_checkpoint/'

# model = resnetkeras.ResNet18((64, 64, 3),200)

# model.build(input_shape=(None,64,64,3))

print(model.summary())

# use_saved_model = False
# checkpoint_filepath = 'resnet_checkpoint/'

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=["accuracy"])

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)

reduceonplateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.001,
    patience=5,
    verbose=1.0,
    mode="auto",
    min_delta=0.005,
    cooldown=0,
    min_lr=0.0
)
print("len valid", len(validation_labels))

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=True,
                              epochs=56,
                              # initial_epoch=50,
                              callbacks=[ reduceonplateau],
                              workers=6)

# print('\n ',history.keys())
# input()

loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=1)
print(loss)
print(acc)