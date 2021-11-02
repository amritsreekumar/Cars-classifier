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
from cubsArch1 import cubsArch1
class t4(Model):

    def __init__(self, pretrained,embedding=512, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.old = pretrained
        self.fc = Dense(embedding)

    def call(self, inputs):
        out = inputs
        for old_layers in self.old.layers[:-1]:
            print(old_layers)
            out = old_layers(out)
            # print(res_block.name)
            # print(out.shape)
      
        out = self.fc(out)
        return out
old = cubsArch1(200)

old.build(input_shape=(100, 64, 64, 3))

# print(old.summary())
model = t4(old)

model.build(input_shape=(100, 64, 64, 3))

print(model.summary())
