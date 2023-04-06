import numpy as np
from network import *

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

# loss function and its derivative
def mse(y_real, y_actual):
    return (np.power(y_real-y_actual, 2))/y_real.size

def mse_derivative(y_real, y_actual):
    return 2*(y_actual-y_real)/y_real.size

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_derivative))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_derivative))

net.use_loss_function(mse, mse_derivative)
net.fit(x_train, y_train, 1000, 0.1)

out = net.predict(x_train)
print(out)