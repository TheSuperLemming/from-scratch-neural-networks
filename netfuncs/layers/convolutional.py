import numpy as np


class CONV:
    """ Convolutional layer
    """

    def __init__(self, kernels, filter_size, activation, pad=0, stride=1, trainable=True):
        """ Initialise the layer
        :param kernels: number of filter kernels
        :param filter_size: filter filter_size
        :param activation: activation function
        :param pad: zero-padding depth
        :param stride: convolution stride length
        :param trainable: set layer trainable (True) or fixed (False)
        """
        self.num = None  # layer number
        self.kernels = kernels
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride
        self.activation = activation
        self.parameters = {'W': None, 'b': None}  # layer parameters (W: weights, b: biases)
        self.Z = None  # linear output (intermediate)
        self.A = None  # layer activation
        self.grads = {'dA': None, 'dZ': None, 'dW': None, 'db': None}  # gradient descent cache
        self.trainable = trainable

    def compile(self, num, input_size, eps=0.01):
        """ Initialise weights and biases, W, b
        :param num: layer number
        :param input_size: image height, width, channels
        :param eps: stddev of W initialisation values
        :return output_size: layer output size
        """
        h, w, c = input_size
        self.num = num
        self.parameters['W'] = np.array(eps * np.random.randn(self.filter_size, self.filter_size, c, self.kernels), dtype=float)
        self.parameters['b'] = np.zeros((1, 1, 1, self.kernels))

        n_h = np.floor((h + 2 * self.pad - self.filter_size) / self.stride + 1).astype('int')
        n_w = np.floor((w + 2 * self.pad - self.filter_size) / self.stride + 1).astype('int')
        output_size = (n_h, n_w, self.kernels)

        return output_size

    def forward_prop(self, A_prev):
        """ Perform forward propagation step
        :param A_prev: activations from previous layer (input_filter_size, m)
        """
        # Pad A_prev
        h, w, c, m = A_prev.shape  # input array height, width, channels, samples
        A_pad = np.zeros((h+2*self.pad, w+2*self.pad, c, m))
        A_pad[self.pad:-self.pad, self.pad:-self.pad, :, :] = A_prev

        # Initialise convolution output array Z
        n_h = np.floor((h + 2*self.pad - self.filter_size)/self.stride + 1).astype('int')
        n_w = np.floor((w + 2*self.pad - self.filter_size)/self.stride + 1).astype('int')
        self.Z = np.zeros((n_h, n_w, self.kernels, m))

        # Loop through filter positions
        # Reshape W, b, A to perform convolution over all kernels and samples in single pass
        # TODO: don't think this works for images with >1 channel
        W = np.reshape(self.parameters['W'], (-1, self.kernels)).T
        b = np.reshape(self.parameters['b'], (-1, self.kernels)).T
        for i in range(0, n_h-self.filter_size, self.stride):
            for j in range(0, n_w-self.filter_size, self.stride):
                a = np.reshape(A_prev[i:i+self.filter_size, j:j+self.filter_size, :, :], (-1, m))
                self.Z[i, j, :, :] = W @ a + b
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


class FLATTEN:
    """ Flatten output from a CONV layer for input to an FC layer
    """

    def __init__(self):
        """ Initialise the layer
        """
        self.num = None  # layer number
        self.A = None  # layer activation
        self.grads = {'dA': None}  # gradient descent cache
        self.shape = None  # array shape cache
        self.trainable = False

    def compile(self, num, input_size):
        """ Parse array sizes for reshape
        :param num: layer number
        :param input_size: size of input array
        """
        self.num = num
        self.shape = input_size
        output_size = np.prod(input_size)

        return output_size

    def forward_prop(self, A_prev):
        """ Flatten input array
        """
        m = A_prev.shape[-1]
        self.A = np.reshape(A_prev, (-1, m))

    def back_prop(self, *args):
        """ Reshape gradient array
        """
        dA_prev = np.reshape(self.grads['dA'], self.shape+(-1,))

        return dA_prev
