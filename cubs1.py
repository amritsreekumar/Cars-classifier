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
        # self.softmax = Activation(activations.softmax)
        self.merge = Add()
        self.sigmoid = Activation(activations.sigmoid)
        self.multiply = Multiply()

    def call(self, x):
        ##BxCxHxW
        # print("Input to", self.name,x.shape)
        x0 = self.gb(x)
        # x0_0 = tf.reshape(x0, (x0.shape[0], 1, x0.shape[1]))
        x0_0 = tf.reshape(x0, (k.shape(x0)[0], 1, k.shape(x0)[1]))

        # x0_0 = tf.keras.layers.Reshape((x0.shape[0], 1, x0.shape[1]))(x0)
        # print("Shape after GB ", x0_0.shape)
        ## parallel dense layers
        ##Bx1xN
        x1 = self.dn1(x0_0)
        # print(type(x1))
        # print("dadassda")
   
        # x2 = tf.reshape(x1, (x1.shape[0], 1, x1.shape[1]))
        # print("pppppppppppppppppppppp")
        x3 = self.dn2(x0_0)
        # x4 = tf.reshape(x3, (x3.shape[0], 1, x3.shape[1]))
        x4 = self.dn3(x0_0)
        # x6 = tf.reshape(x5, (x5.shape[0], 1, x5.shape[1]))
        # print("x4 shape", x4.shape)
        
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

        Similarity_matrix = tf.matmul(x1, x3, transpose_a=True)
        # print("dasdsadasdasasd")
        # print("shape similarity mat", Similarity_matrix.shape)
        ##BxNxN
        # softmax = self.softmax(Similarity_matrix)
        # softmax = Activation(activations.softmax(Similarity_matrix, axis=-1))
        softmax = tf.keras.activations.softmax(Similarity_matrix, axis=1)

        # print("Shape after softmax", softmax.shape)
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

        ##last dense; assuming C = input data channles, and channels is 1st intem in shape list
        
        sf_dot_x3 = tf.matmul(x4, softmax)
        # print("shape sf_dot_x3", sf_dot_x3)
        x7 = self.dn4(sf_dot_x3)
        # print("x7 shape", x7.shape)
        # x8 = tf.reshape(x7, (x7.shape[0], 1, x7.shape[1]))
        # print("dasdsadasdasasd")
        ##Add ouput of Dense layer and GAP
        x9 = self.merge([x7, x0_0])
        # print("x9 shape", x9.shape)
        ##Bx1xC
        x10 = self.sigmoid(x9)
        # print("after sigmoid", x10.shape)
        x10 = tf.expand_dims(x10, axis=2)
        # print("after sigmoid", x10.shape)
        ##Multiply
        x11 = self.multiply([x, x10])
        # print("x11 shape ", x11.shape)

        return x11
