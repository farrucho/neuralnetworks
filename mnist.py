from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from network import *
from keras.datasets import mnist
from keras.utils import np_utils

# load mnist dataset
# (imagem_training, label), (imagem_test, label)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape para uma matriz 1xN, N é uma matriz imagem quadrada 28x28
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255 # ter valores entre 0 e 1

# se for 3 fica y = [0 0 0 1 0 0 0 0 0 0]
y_train = np_utils.to_categorical(y_train)


# fazer o mesmo para a test data
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)


# Network 28*28 -> 100 -> 50 -> 10

net = Network()
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_derivative))
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_derivative))
net.add(FCLayer(50, 10))
net.use_loss_function(mse, mse_derivative)

net.read()

net.fit(x_train, y_train, 3, 0.1)

# test on 3 samples
out = net.predict(x_train[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_train[0:3])

net.save()