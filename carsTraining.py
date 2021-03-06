from re import L
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models, optimizers, activations
from tensorflow.keras.losses import categorical_crossentropy
import os
import tensorflow as tf
from Dataloader import DataGeneratorCars
from scipy.io import loadmat
import cubsArch1
from cubsArch1 import cubsArch1
import itertools
if __name__ == "__main__":
    path_to_train_images = './car_ims'
    #labelsTest = {'00001.jpg': 1, '00002.jpg': 2, '00003.jpg': 3, '00004.jpg': 4, '00005.jpg': 5}

    path_to_labels = './cars_annos.mat'
    #path_to_classes = './devkit/cars_meta.mat'
    #print(path_to_classes)

    mat_train = loadmat(path_to_labels)
    labels = {}
    labelstest = {}

    for example in mat_train['annotations'][0]:
        label = example[5][0]
        image = example[0][0].split('/')[1]
        if label < 99:
            labels[image] = label
        else:
            labelstest[image] = label

    


    model = cubsArch1(98)

    #model.build(input_shape=(100, 64, 64, 3))
    ##model compilation
    use_saved_model = False

    checkpoint_filepath = 'resnetcubs_cars_checkpoint/'

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                metrics=["accuracy"])

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

    print(len(labels))
    X_train = dict(list(labels.items())[:7500])
    X_valid = dict(list(labels.items())[7500:])

    print( "Training labels size = ", len(X_train))
    print( "Validation labels size = ", len(X_valid))
    print( "Test labels size = ", len(labelstest))

    trainGenerator = DataGeneratorCars(
        labels=X_train, rescale=1. / 255, path_to_train=path_to_train_images, val=False, flag = 1,
        batch_size=16, target_size=(256, 256), channels=3, n_classes=98)

    ##Does batch size have to be exact divisor of training samplesx
    validGenerator = DataGeneratorCars(
        labels=X_valid, rescale=1. / 255, path_to_train=path_to_train_images, val=True, flag =0,
        batch_size=16, target_size=(256, 256), channels=3, n_classes=98)

    testGenerator = DataGeneratorCars(
        labels=labelstest, rescale=1. / 255, path_to_train=path_to_train_images, val=True, flag = 0,
        batch_size=16, target_size=(256, 256), channels=3, n_classes=98)

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

    #history = model.fit(trainGeenrator, steps_per_epoch=2000, epochs=10)
    #history

    model.fit_generator(generator=trainGenerator,
                              validation_data=validGenerator,
                              use_multiprocessing=False,
                              epochs=500,
                              # initial_epoch=50,
                              callbacks=[model_checkpoint_callback, reduceonplateau],
                              workers=6)

    #loss, acc = model.evaluate_generator(testGenerator, steps=3, verbose=1)
    #print(acc)  
    #print(loss)