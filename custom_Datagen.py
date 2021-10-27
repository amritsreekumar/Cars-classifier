import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class Datagen(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, label_ids, val_dict, val, flag = 1, batch_size=32, n_classes=200, dim=(64, 64), n_channels=3,
                 shuffle=True):
        self.val = val
        self.dim = dim
        self.label_ids = label_ids
        self.val_dict = val_dict  # to map between validation images and their corresponding classes
        self.flag = flag #set value = 1, when 30 percent of original images needs to be passed for training. value = 0, all images augmented
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
            # vertical_flip=True,
            # brightness_range=[0.2,1.2],
            fill_mode='nearest'
        )

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        ##Select a set of
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if self.val == False:
            list_IDs_temp = [k for k in indexes]  # Generate data
        else:
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # print(list_IDs_temp)
        X, y = self.__data_generation(list_IDs_temp)  # to be implemented
        return X, y

    def on_epoch_end(self):
        if self.val == True:
            self.indexes = np.arange(len(self.list_IDs))
        else:
            self.indexes = self.list_IDs
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # to load the data from the indices we give it
    # indices should be given in the main script
    def __data_generation(self, list_IDs_temp):
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        if self.val == True:
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                # X[i,] = load_img('tiny-imagenet-200/val/images/' + ID)
                X[i,] = load_img('/home/rahulnai/Workspace/Cars-classifier/tiny-imagenet-200/val/images/' + ID)
                # Store class
                labelid = self.val_dict[ID]
                new_label = self.label_ids[labelid]
                y[i] = new_label
            X = X / 255.
            # X_transformed = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)

            # return next(X_transformed), tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
            return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        elif self.val == False:
            for i, filedata in enumerate(list_IDs_temp):
                # Store sample
                folderName, filename = os.path.basename(filedata).split('.')[0].split('_')
                X[i,] = load_img(
                    '/home/rahulnai/Workspace/Cars-classifier/tiny-imagenet-200/train/' + folderName + '/images/' + folderName + '_' + str(
                        filename) + '.JPEG')
                # Store class
                labelid = self.label_ids[folderName]
                y[i] = labelid
            X = X / 255.

            if self.flag == 1:
                length = len(X)
                length = int(0.7*length)
                part_1_X = X[:length]
                part_2_X = X[length:]
                X_transformed = self.augmentor.flow(part_1_X, batch_size=length, shuffle=False)
                X_transformed = X_transformed.next()
                New_set = np.concatenate((X_transformed,part_2_X),axis=0)
            else:
                X_transformed = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)
                New_set = next(X_transformed)
            
            return New_set, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)