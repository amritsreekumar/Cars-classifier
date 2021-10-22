import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, \
    Dense, Input, Activation, MaxPool2D

from tensorflow.keras import activations
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Layer, Add, \
    Multiply
import numpy as np


class cubs1(object):
    def __init__(self, input, denseNN):
        ##Output from a res block BxCxHxW
        self.input = input
        self.C = input.shape[1]  ##expecting to parse channels
        self.gb = GlobalAveragePooling2D()
        self.dn1 = Dense(units=denseNN)
        self.dn2 = Dense(units=denseNN)
        self.dn3 = Dense(units=denseNN)
        self.dn4 = Dense(units=self.C)
        ##dot(dn2T, dn3) nx1 1xn = nxn
        self.softmax = Activation(activations.softmax)
        self.merge = Add()
        self.sigmoid = Activation(activations.sigmoid)
        self.multiply = Multiply()

    def call(self):
        ##BxCxHxW
        x = self.gb(x)
        ## parallel dense layers
        x1 = self.dn1(x)
        x2 = self.dn2(x)
        x3 = self.dn3(x)

        Similarity_matrix = np.dot(x1.T, x2)
        sf = self.softmax(Similarity_matrix)
        ##doubtful
        sf_plus_x3 = self.merge(sf, x3)
        ##last dense; assuming C = input data channles, and channels is 1st intem in shape list

        x4 = self.dn4(sf_plus_x3)
        ##Add ouput of Dense layer and GAP
        x4 = self.merge(x4, x)
        ##Bx1xC
        x4 = self.sigmoid(x4)

        ##Multiply
        x4 = self.multiply(x, x4)

        return x4
