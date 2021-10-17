from skeleton2 import Datagen
from resnet import ResNet18
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

if __name__ == "__main__":
    model = ResNet18(10)
    model.build(input_shape=(None, 32, 32, 3))
    # use categorical_crossentropy since the label is one-hot encoded
    from keras.optimizers import SGD

    # opt = SGD(learning_rate=0.1,momentum=0.9,decay = 1e-04) #parameters suggested by He [1]
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
