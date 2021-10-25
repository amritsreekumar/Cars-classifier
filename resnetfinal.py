# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
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
import tensorflow.keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

import Datagen as Datagen
import resnetkeras


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out
class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2,
                          self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    ### ETA
    class_ids = open('./tiny-imagenet-200/wnids.txt', "r")
    class_ids = class_ids.readlines()
    label_ids = {}
    for i in range(len(class_ids)):
        label_ids[class_ids[i].split('\n')[0]] = i

    ####Full path to all images
    train_paths = glob.glob('./tiny-imagenet-200/train/**/*.JPEG', recursive=True)

    train_labels = []
    for i in train_paths:

        train_labels.append(os.path.basename(i))
        


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

    np.random.shuffle(validation_images)
    
    X_valid = validation_images[:8000]
    X_test = validation_images[8000:]

    ##pass folder names (train labels), foldername to class mappings(label ids) 
    training_generator = Datagen.Datagen(train_labels, label_ids, val_dict = None, val = False, batch_size=200, flag=1)
    validation_generator = Datagen.Datagen(X_valid, label_ids, val_dict, val = True, batch_size=200, flag=0)
    test_generator = Datagen.Datagen(X_test, label_ids, val_dict, val = True, batch_size=200, flag=0)


    model = ResNet18(200)

    model.build(input_shape=(None, 64, 64, 3))

    print(model.summary())

    use_saved_model = False
    checkpoint_filepath = 'resnet_checkpoint/'

    if use_saved_model:

        model = tf.keras.models.load_model(checkpoint_filepath)
        loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=0)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


    # model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"],)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.1), loss = 'categorical_crossentropy', 
                metrics = ["accuracy"])

  
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

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
                        use_multiprocessing=False,
                        epochs = 100,
                        # initial_epoch=50,
                        callbacks = [model_checkpoint_callback, reduceonplateau] ,
                        workers = 6)
    
    # print('\n ',history.keys())
    # input()

    loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=1)
    print(loss)
    print(acc)