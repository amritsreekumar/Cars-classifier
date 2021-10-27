import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import datasets,models,layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Activation, Multiply
# from keras.models import Sequential
# from tensorflow.keras import Model
from tensorflow.keras import activations
import tensorflow.keras.backend as k

class cubs2(Model):
    def __init__(self, **kwargs):
        ##Output from a res block BxCxHxW
        super().__init__(**kwargs)
        # self.input = input
        # self.C = channels  ##expecting to parse channels
        # self.gb = GlobalAveragePooling2D()
        self.conv1 = Conv2D(filters=1, kernel_size=(1,1))
        self.conv2 = Conv2D(filters=1, kernel_size=(1,1))
        self.conv3 = Conv2D(filters=1, kernel_size=(1,1))
        
    
        self.softmax = Activation(activations.softmax)
       
        self.sigmoid = Activation(activations.sigmoid)
        self.multiply = Multiply()
    def call(self, x):
        x0 = self.conv1(x)
        x0 = tf.reshape(x0, (k.shape(x0)[0], 1, k.shape(x0)[1]*k.shape(x0)[2]))
        x1 = self.conv2(x)
        x1 = tf.reshape(x1, (k.shape(x1)[0], 1, k.shape(x1)[1]*k.shape(x1)[2]))
        x2 = self.conv3(x)
        x2 = tf.reshape(x2, (k.shape(x2)[0], 1, k.shape(x2)[1]*k.shape(x2)[2]))

        Similarity_matrix = tf.matmul(x0, x1, transpose_a=True)

        softmax = self.softmax(Similarity_matrix)

        features = tf.matmul(x2, softmax)

        sigmoidOut = self.sigmoid(features)

        sig_reshped_h_w = tf.reshape(sigmoidOut, (k.shape(sigmoidOut)[0], k.shape(x)[1], k.shape(x)[2], 1 ))

        out = self.multiply([sig_reshped_h_w, x])

        return out

model = cubs2()
model.build(input_shape=(100, 64, 64, 300))
model.summary()



    
    