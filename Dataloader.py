import numpy as np
import keras
import os
#import matplotlib.pyplot as plt
from PIL import Image as pil_image
import io
from keras_preprocessing.image import ImageDataGenerator
'''
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

'''
class DataGeneratorCars(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, labels, path_to_train, rescale=1./255, batch_size=32, target_size=(256, 256), channels=1,
                 n_classes=10, shuffle=True):
        ##target_size - if row and col shapes are different then check how pil image loads data
        ### as target shape is checked against loaded pil image.shape
        #self.dim = dim
        self.target_size = target_size
        self.path_to_train = path_to_train
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = self._read_from_directory()
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.rescale = rescale
        self.indexes = np.arange(len(self.list_IDs))
        self.dtype = 'float32'
        self._PIL_INTERPOLATION_METHODS = {
            'nearest': pil_image.NEAREST,
            'bilinear': pil_image.BILINEAR,
            'bicubic': pil_image.BICUBIC,
        }

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def _read_from_directory(self):
        '''

        :param path_to_train: path to training folder, does not support subdirectory classes
        :return: list of filenames
        '''
        if self.path_to_train:
            files = [file for file in os.listdir(self.path_to_train)]
        return files

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        #print(index)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _rescale(self, img):
        if self.rescale:
            return img * self.rescale
        else:
            return img

    def __data_generation(self, list_IDs_temp, resample='nearest'):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization


        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #X = np.empty((self.batch_size, target_size ,self.n_channels))
        batch_x = np.zeros((len(list_IDs_temp),) + self.target_size + (self.channels, ), dtype=self.dtype)
        batch_y = np.zeros(len(list_IDs_temp), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            #with open()
            #X[i,] = pil_image.open(self.path_to_train + '/' + ID, format='jpeg')
            with open(self.path_to_train + '/' + ID, 'rb') as f:
                img = pil_image.open(io.BytesIO(f.read()))
                if img.size != self.target_size:
                    img = img.resize(self.target_size, self._PIL_INTERPOLATION_METHODS[resample])
                    img = np.asarray(img, dtype=self.dtype)
            batch_x[i] = self._rescale(img)



            # Store class
            batch_y[i] = self.labels[ID]

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)
        #return batch_x, batch_y
