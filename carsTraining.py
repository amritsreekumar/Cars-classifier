import numpy as np
from keras import layers
from keras import models, optimizers, activations
from keras.losses import categorical_crossentropy
import os

from scipy.io import loadmat

from Dataloader import DataGeneratorCars

if __name__ == "__main__":
    path_to_train_images = '/Users/rahulnair/Desktop/personal/UB/courses/cse 673 Comp Vision/Assign2_3/carsData/train_sample'
    #labelsTest = {'00001.jpg': 1, '00002.jpg': 2, '00003.jpg': 3, '00004.jpg': 4, '00005.jpg': 5}

    path_to_labels = '/Users/rahulnair/Desktop/personal/UB/courses/cse 673 Comp Vision/Assign2_3/TrainLabels/devkit/cars_train_annos.mat'
    path_to_classes = '/Users/rahulnair/Desktop/personal/UB/courses/cse 673 Comp Vision/Assign2_3/TrainLabels/devkit/cars_meta.mat'

    mat_train = loadmat(path_to_labels)
    labels = {}

    for example in mat_train['annotations'][0]:
        label = example[-2][0][0]
        image = example[-1][0]
        labels[image] = label
    #
    print( "Training labels size = ", len(labels))


    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    #model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(196, activation='softmax'))

    model.summary()

    ##model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['acc'])

    trainGeenrator = DataGeneratorCars(
        labels=labels, rescale=1. / 255, path_to_train=path_to_train_images,
        batch_size=5, target_size=(256, 256), channels=3, n_classes=196)

    ##Does batch size have to be exact divisor of training samples

    validGenerator = DataGeneratorCars(
        labels=labels, rescale=1. / 255, path_to_train=path_to_train_images,
        batch_size=5, target_size=(256, 256), channels=3, n_classes=196)

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





