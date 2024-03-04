from supportfunctions import check_gradient
import matplotlib.pyplot as plt
import numpy as np
import copy


class Model:
    """ Model class
    """

    def __init__(self, layers, cost_fun, optimiser, metrics=[]):
        """ Initialise the model
        :param layers: list of layer objects
        :param cost_fun: cost function object
        :param metrics: list of metric objects
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.cost_fun = cost_fun
        self.optimiser = optimiser
        self.metrics = metrics

    def compile(self, size):
        """ Initialise model layers
        :param size: number of elements in data sample
        """
        for l, layer in enumerate(self.layers):
            size = layer.compile(l, size)
            if layer.trainable:
                self.optimiser.compile(l, layer.parameters.keys())

    def train(self, X, Y, num_epochs=100, plot_cost=False, check_grads=()):
        """ Train the model
        :param X: input data array (num features x num samples)
        :param Y: label data array (num classes x num samples
        :param num_epochs: number of training epochs
        :param plot_cost: boolean flag to switch cost plot
        :param check_grads: numerical gradient descent check at epochs in list
        """
        for n in range(num_epochs):

            # Forward pass
            self.layers[0].forward_prop(X)
            for l in range(1, self.num_layers):
                self.layers[l].forward_prop(self.layers[l-1].A)

            # Compute cost
            J = self.cost_fun.cost(self.layers[-1].A, Y)
            print("Epoch {}: J={}".format(n, J))
            if plot_cost:
                # Plot cost function
                plt.figure("cost_plot")
                plt.plot(n, J, marker='o', color='r')
                plt.show(block=False)
                plt.pause(0.00001)

            # Back pass
            self.layers[-1].grads['dZ'] = self.cost_fun.dZ(self.layers[-1].A, Y)
            for l in range(self.num_layers-1, 0, -1):
                dA_prev = self.layers[l].back_prop(self.layers[l - 1].A, self.optimiser)
                self.layers[l - 1].grads['dA'] = dA_prev
            self.layers[0].back_prop(X, self.optimiser)

            if n in check_grads:
                clone = copy.deepcopy(self)
                check_gradient(clone, X, Y)

    def predict(self, X):
        """ Generate a set of predictions
        :param X: input data
        :return Y: output predictions
        """
        self.layers[0].forward_prop(X)
        for l in range(1, self.num_layers):
            self.layers[l].forward_prop(self.layers[l - 1].A)

        i_pred = np.argmax(self.layers[-1].A, axis=0)
        Y_pred = np.zeros(self.layers[-1].A.shape)
        for j, i in enumerate(i_pred):
            Y_pred[i, j] = 1

        return Y_pred

    def evaluate(self, Y_pred, Y):
        """ Evaluate performance metrics
        :param Y_pred: model predictions
        :param Y: ground truth label data
        :return indices: dict of correctly and incorrectly predicted sample indices
        :return metrics: dict of evaluated metrics
        """
        correct_preds = np.sum(Y_pred * Y, axis=0)
        indices = {'correct': [], 'incorrect': []}
        for i, pred in enumerate(correct_preds):
            indices['correct'].append(i) if pred else indices['incorrect'].append(i)

        metrics = {}
        for m in self.metrics:
            metrics[m.name] = m.evaluate(indices)

        return indices, metrics
