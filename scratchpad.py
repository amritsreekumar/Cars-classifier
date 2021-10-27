
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Dense, Conv2D
batch_size, n, m, k = 10, 3, 5, 2
A = tf.Variable(tf.random.normal(shape=(batch_size, n, m)))
B = tf.Variable(tf.random.normal(shape=(batch_size, m, k)))
# tf.matmul(A, B)
c = A @ B
print(c.shape)

# D = tf.Variable(tf.random.normal(shape=(None, 64)))
# print(D.shape)
# D = tf.reshape(D, (None, 1, 64))
# print(D.shape)
# print(type(D))


conv1 = Conv2D(filters=64, kernel_size=(2,2))
den1 = Dense(64)
## input is a placeholder,
input = tf.keras.Input(shape=(64, 64,3), batch_size=100)
convout = conv1(input)
denout = den1(convout)

model = tf.keras.models.Model(input, denout)
model.build(input_shape=(1000, 64, 64, 3))

print(model.summary())

model.compile(loss='mse')
ones = tf.ones(shape=(100, 64,64,3))

model.fit(ones, y=tf.zeros(shape=(100,)))

