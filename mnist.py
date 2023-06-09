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
net.add(FCLayer(28*28, 128))
net.add(ActivationLayer(tanh, tanh_derivative))
net.add(FCLayer(128, 64))
net.add(ActivationLayer(tanh, tanh_derivative))
net.add(FCLayer(64, 10))
net.use_loss_function(mse, mse_derivative)

net.read()

""" net.read()
for j in range(0,100):
    net.fit(x_train, y_train, 5, 0.001)
    net.fit(x_test, y_test, 5, 0.001)
    net.save()
    k=0
    net.read()
    for j in range(len(x_test)):
        out = net.predict(x_test[j])
        if np.argmax(out) != np.argmax(y_test[j]):
            print(f"index: {j}, resultado: {np.argmax(out)}, suposto: {np.argmax(y_test[j])}")
            k += 1
    print("accuracy: " + str(100-100*k/len(x_test)) + "%") """

""" k=0
net.read()
for j in range(len(x_test)):
    out = net.predict(x_test[j])
    if np.argmax(out) != np.argmax(y_test[j]):
        print(f"index: {j}, resultado: {np.argmax(out)}, suposto: {np.argmax(y_test[j])}")
        k += 1
print("accuracy: " + str(100-100*k/len(x_test)) + "%") """
#96.21

""" # test on 3 samples
out = net.predict(x_test[570:573])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[570:573]) """

#PROXIMO FAZER FAZER QUICKDRAW
# DAR CONTROL C NOS ERROS E EPOCHS PARA FAZER GRAFICO E COMPARAR PERFORMANCE
# script automatico para testar em varias networks
