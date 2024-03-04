import numpy as np


class BinaryCrossEntropy:
    """ Binary classifier cost function
    """
    @staticmethod
    def cost(ypred, Y):
        """ Compute the cost
        :param ypred: prediction
        :param Y: data labels
        :return J: evaluated cost
        """
        _, m = Y.shape
        J = -(Y @ np.log(ypred.T) + (1-Y) @ np.log(1-ypred.T)) / m

        return np.squeeze(J)

    @staticmethod
    def dZ(ypred, Y):
        """ Compute dZ to initiate backprop
        :param ypred: prediction
        :param Y: data labels
        :return dZ: partial differential dJ/dZ
        """
        dZ = ypred - Y

        return dZ


class CategoricalCrossEntropy:
    """ Multiclass classifier cost function
    """
    @staticmethod
    def cost(ypred, Y):
        """ Compute the cost
        :param ypred: prediction
        :param Y: data labels
        :return: evaluated cost
        """
        _, m = Y.shape
        J = -np.sum(Y*np.log(ypred)) / m

        return np.squeeze(J)

    @staticmethod
    def dZ(ypred, Y):
        """ Compute dZ to initiate backprop
        :param ypred: prediction
        :param Y: data labels
        :return dZ: partial differential dJ/dZ
        """
        dZ = ypred - Y

        return dZ
