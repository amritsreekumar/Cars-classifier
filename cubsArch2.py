import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# from tensorflow.keras import datasets,models,layers
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
# from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
# from keras.models import Sequential
# from tensorflow.keras import Model

import tensorflow as tf
import cubs1
from cubs2 import cubs2

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
        self.cubs1 = cubs1.cubs1(self.__channels, self.__channels/2)
        self.cubs2 = cubs2()

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
        x = self.cubs1(x)
        x = self.cubs2(x)
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class cubs1cubs2res1(Model):

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
        # self.cub1_1 = cubs1.cubs1(64, 32)
        # self.cub2_1 = cubs2()
        self.res_1_2 = ResnetBlock(64)
        # self.cub1_2 = cubs1.cubs1(64, 32)
        # self.cub2_2 = cubs2()
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        # self.cub1_3 = cubs1.cubs1(128, 64)
        # self.cub2_3 = cubs2()
        self.res_2_2 = ResnetBlock(128)
        # self.cub1_4 = cubs1.cubs1(128, 64)
        # self.cub2_4 = cubs2()
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        # self.cub1_5 = cubs1.cubs1(256, 64)
        # self.cub2_5 = cubs2()
        self.res_3_2 = ResnetBlock(256)
        # self.cub1_6 = cubs1.cubs1(256, 64)
        # self.cub2_6 = cubs2()
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        # self.cub1_7 = cubs1.cubs1(512, 128)
        # self.cub2_7 = cubs2()
        self.res_4_2 = ResnetBlock(512)
        # self.cub1_8 = cubs1.cubs1(512, 128)
        # self.cub2_8 = cubs2()
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        # inputs = keras.Input((64, 64, 3))
        # print("input", inputs.shape)
        out = self.conv_1(inputs)
        # print("conv1_1", out.shape)
        out = self.init_bn(out)
        # print("batchnorm", out.shape)
        out = tf.nn.relu(out)
        # print("relu", out.shape)
        out = self.pool_2(out)
        # print("max pool", out.shape)
        for res_block in [self.res_1_1,
                        
                         self.res_1_2, 
                         
                         self.res_2_1,
                         
                         self.res_2_2,
                        
                         self.res_3_1,
                         
                         self.res_3_2,
                         
                         self.res_4_1,
                         
                         self.res_4_2
                         ]:
            
            out = res_block(out)
            # print(res_block.name)
            # print(out.shape)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

model = cubs1cubs2res1(200)

model.build(input_shape=(100, 64, 64, 3))

print(model.summary())
