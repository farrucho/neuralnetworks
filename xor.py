import numpy as np
from network import *


x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_derivative))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_derivative))

net.read()

net.use_loss_function(mse, mse_derivative)
net.fit(x_train, y_train, 100000, 0.1)
out = net.predict(x_train)
print(out)
net.save()