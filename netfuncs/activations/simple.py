import numpy as np


class Identity:
    """ Identity function for test purposes
    """
    @staticmethod
    def f(z):
        return z

    @staticmethod
    def df():
        return 1


class Sigmoid:
    """ Sigmoid activation function
    """
    @staticmethod
    def f(z):
        return 1 / (1+np.exp(-z))

    def df(self, z):
        s = self.f(z)
        return s * (1-s)


class Tanh:
    """ Tanh activation function
    """
    @staticmethod
    def f(z):
        e_plus = np.exp(z)
        e_minus = np.exp(-z)
        return (e_plus - e_minus) / (e_plus + e_minus)

    def df(self, z):
        return 1 - self.f(z)**2


class ReLU:
    """ Rectified Linear Unit (ReLU) activation function
    """
    @staticmethod
    def f(z):
        return np.maximum(z, 0)

    def df(self, z):
        return (z > 0).astype(int)


class LeakyReLU:
    """ Leaky ReLU activation function
    """
    def __init__(self, k=0.01):
        self.k = k  # scale factor for z < 0

    def f(self, z):
        return np.maximum(z, self.k*z)

    def df(self, z):
        return np.maximum((z > 0).astype(int), self.k)


class Softmax:
    """ Softmax activation function
    """
    @staticmethod
    def f(z):
        t = np.exp(z)
        return t / np.sum(t, axis=0)

    @staticmethod
    def df():
        raise(Exception("No derivative defined for costfunctions.Softmax, use costfunction.dZ to initiate backprop"))


