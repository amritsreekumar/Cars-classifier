from re import L
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models, optimizers, activations
from tensorflow.keras.losses import categorical_crossentropy
import os
import tensorflow as tf
from Dataloader import DataGeneratorCars
from scipy.io import loadmat
import cubsArch2
from cubsArch2 import cubs1cubs2res1

if __name__ == "__main__":
    path_to_train_images = './cars_train'
    #labelsTest = {'00001.jpg': 1, '00002.jpg': 2, '00003.jpg': 3, '00004.jpg': 4, '00005.jpg': 5}

    path_to_labels = './carsmeta/cars_train_annos.mat'
    path_to_classes = './carsmeta/cars_meta.mat'
    #print(path_to_classes)

    mat_train = loadmat(path_to_labels)
    labels = {}
    labelstest = {}

    for example in mat_train['annotations'][0]:
        label = example[-2][0][0]
        if label < 99:
            image = example[-1][0]
            labels[image] = label
        else:
            image = example[-1][0]
            labelstest[image] = label

    


    print( "Training labels size = ", len(labels))
    print( "Test labels size = ", len(labelstest))

    model = cubs1cubs2res1(98)

    model.build(input_shape=(100, 64, 64, 3))
    ##model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['acc'])

    trainGeenrator = DataGeneratorCars(
        labels=labels, rescale=1. / 255, path_to_train=path_to_train_images,
        batch_size=10, target_size=(256, 256), channels=3, n_classes=98)

    ##Does batch size have to be exact divisor of training samples

    testGenerator = DataGeneratorCars(
        labels=labelstest, rescale=1. / 255, path_to_train=path_to_train_images,
        batch_size=10, target_size=(256, 256), channels=3, n_classes=98)

    # validDataGenerator = validationGener.flow_from_directory(
    #     validaDir,
    #     target_size=(150, 150),
    #     class_mode='binary',
    #     batch_size=20
    # )

    # history = model.fit_generator(trainGeenrator, steps_per_epoch=100, epochs=1,
    #                               validation_data=validDataGenerator,
    #                               validation_steps=50)

    # history = model.fit_generator(trainGeenrator, steps_per_epoch=2000, epochs=10)

    history = model.fit(trainGeenrator, steps_per_epoch=2000, epochs=10)
    #history