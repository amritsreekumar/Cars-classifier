import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# from tensorflow.keras import datasets,models,layers
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
# from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Activation, Multiply
# from keras.models import Sequential
# from tensorflow.keras import Model
from tensorflow.keras import activations

class cubs1(Model):
    def __init__(self, channels, denseNN, **kwargs):
        ##Output from a res block BxCxHxW
        super().__init__(**kwargs)
        # self.input = input
        self.C = channels  ##expecting to parse channels
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

    def call(self, x):
        ##BxCxHxW
        print("Input to", self.name,x.shape)
        x0 = self.gb(x)
        print("Shape after GB ", x0.shape)
        ## parallel dense layers
        ##Bx1xN
        x1 = self.dn1(x0)
        print(x1.shape,"x1")
        x2 = self.dn2(x0)
        x3 = self.dn3(x0)

        # Similarity_matrix = np.dot(np.array(x1).T, np.array(x2))
        # print("shape x1", x1.shape)
        
        # tmp = []
        # try:
        #     for i in range(x1.shape[0]):
        #         tmptmp = tf.matmul( tf.reshape(x1[i], (x1[i].shape[0], 1)) , tf.reshape(x2[i], (1, x1[i].shape[0])) ) 
        #         tmp.append(tmptmp)
        #     Similarity_matrix = tf.stack(tmp)
        #     print("shape similarity mat", Similarity_matrix.shape)
        # except TypeError:
        #     Similarity_matrix = keras.Input(shape=(100, 64, 64))
        
        Similarity_matrix = tf.matmul(tf.reshape(x1, (1, x1.shape[1])), x2, transpose_a=True)

        ##BxNxN
        softmax = self.softmax(Similarity_matrix)
        print("Shape after softmax", softmax.shape)
        ##doubtful
        ##Bx1xN
        # tmp = []
        # try:
        #     for i in range(x3.shape[0]):
        #         tmptmp = tf.matmul( tf.reshape(x3[i], (1, x3[i].shape[0])) , softmax[i] )
        #         tmp.append(tmptmp)
        #     sf_dot_x3 = tf.stack(tmp)
        #     print("sf_dot_x3", sf_dot_x3.shape)
        # except TypeError:
        #     sf_dot_x3 = keras.Input(shape=(100, 1, 64))
        sf_dot_x3 = tf.matmul(tf.reshape(x3, (1, x3.shape[0])), softmax, transpose_a=True)

        ##last dense; assuming C = input data channles, and channels is 1st intem in shape list

        x4 = self.dn4(sf_dot_x3)
        ##Add ouput of Dense layer and GAP
        x4 = self.merge([x4, x])
        print("x4 shape", x4.shape)
        ##Bx1xC
        x4 = self.sigmoid(x4)
        print("after sigmoid", x4.shape)

        ##Multiply
        x5 = self.multiply([x, x4])
        print("Out shape ", x5.shape)

        return x5

model = cubs1(3, 64)
model.build(input_shape=(100, 64, 64, 3))
model.summary()