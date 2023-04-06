import numpy as np
# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65


class FCLayer():
    def __init__(self, N_inputs, N_outputs): # layer são 2 layers
        self.weights = np.random.rand(N_inputs,N_outputs) - 0.5
        self.bias = np.random.rand(1, N_outputs) - 0.5
        # assim os valores tao entre -0.5 e 0.5 pos/neg

    def mudar_param(self, weights, bias, input, output):
        self.weights = weights
        self.bias = bias
        self.input = input
        self.output = output

    def forward_propagation(self, input_data): # input data, valor dos nodes da esquerda recebidos
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error

        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * bias_error

        return input_error


class ActivationLayer():
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.weights = []
        self.bias = []

    def mudar_param(self, weights, bias, input, output):
        self.weights = weights
        self.bias = bias
        self.input = input
        self.output = output

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.activation_derivative(self.input)


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None # y,output

    def add(self, layer):
        self.layers.append(layer)

    def use_loss_function(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)

        return output

    def fit(self, x_lista, y_lista, epochs, learning_rate):
        samples = len(x_lista)
        
        for i in range(epochs):
            err = 0
            
            for j in range(samples): # rodar nas samples
                # forward propagation
                output = self.predict(x_lista[j])

                err += self.loss(y_lista[j],output)
                
                # backpropagation
                error = self.loss_derivative(y_lista[j],output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
        
            err = np.sum(err) / samples * 100
            print('epoch: ' + str(i+1) + "  error = " + str(err) + "%")

    def save(self):
        print("salvando dados")
        f = open("network.txt", "w")
        f.write("")
        f.close()
        f = open("network.txt", "a")
        for layer in self.layers:
            f.write(str(np.array(layer.weights).tolist()).replace("\n","") + "\n" + str(np.array(layer.bias).tolist()).replace("\n","") + "\n" + str(np.array(layer.input).tolist()).replace("\n","") + "\n" + str(np.array(layer.output).tolist()).replace("\n","") + "\n")
        f.close()
        print("guardado dados")

    def read(self):
        f = open("network.txt", "r")
        texto = f.read()
        f.close()
        lista_tudo = (texto.replace("None", "[0]")).split("\n")
        import json
        for j in range(len(self.layers)):
            self.layers[j].mudar_param(np.array(json.loads(lista_tudo[j*4])), np.array(json.loads(lista_tudo[j*4+1])), np.array(json.loads(lista_tudo[j*4+2])), np.array(json.loads(lista_tudo[j*4+3])))
        print("dados lidos")
            

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



    