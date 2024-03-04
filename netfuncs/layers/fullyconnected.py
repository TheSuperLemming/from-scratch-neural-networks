import numpy as np


class FC:
    """ Fully connected layer
    """

    def __init__(self, size, activation, trainable=True):
        """ Initialise the layer
        :param size: number of neurons in layer
        :param activation: activation function
        :param trainable: set layer trainable (True) or fixed (False)
        """
        self.num = None  # layer number
        self.size = size
        self.activation = activation
        self.parameters = {'W': None, 'b': None}  # layer parameters (W: weights, b: biases)
        self.Z = None  # linear output (intermediate)
        self.A = None  # layer activation
        self.grads = {'dA': None, 'dZ': None, 'dW': None, 'db': None}  # gradient descent cache
        self.trainable = trainable

    def compile(self, num, input_size, eps=0.01):
        """ Initialise weights and biases, W, b
        :param num: layer number
        :param input_size: number of neurons in previous layer
        :param eps: stddev of W initialisation values
        :return size: layer output size
        """
        self.num = num
        self.parameters['W'] = np.array(eps * np.random.randn(self.size, input_size), dtype=float)
        self.parameters['b'] = np.zeros((self.size, 1))

        return self.size

    def forward_prop(self, A_prev):
        """ Perform forward propagation step
        :param A_prev: activations from previous layer (input_size, m)
        """
        self.Z = self.parameters['W'] @ A_prev + self.parameters['b']
        self.A = self.activation.f(self.Z)

    def back_prop(self, A_prev, optimiser):
        """ Perform back propagation step
        :param A_prev: activations from previous layer
        :param optimiser: gradient descent optimiser
        :return dA_prev: dA calculated for layer l-1
        """

        if self.grads['dZ'] is None:
            self.grads['dZ'] = self.grads['dA'] * self.activation.df(self.Z)

        m = self.grads['dZ'].shape[-1]

        self.grads['dW'] = self.grads['dZ'] @ A_prev.T / m
        self.grads['db'] = np.sum(self.grads['dZ'], axis=1, keepdims=True) / m
        dA_prev = self.parameters['W'].T @ self.grads['dZ']

        if self.trainable:
            self.parameters = optimiser.update_parameters(self.num, self.parameters, self.grads)

        return dA_prev
