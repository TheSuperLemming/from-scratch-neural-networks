class Base:
    """ Base optimiser class
    """
    def __init__(self, alpha):
        """ Initialise the optimiser
        :param alpha: learning rate
        """
        self.alpha = alpha

    def compile(self, *args, **kwargs):
        """ Base compile method
        """


class GradDescent(Base):
    """ Basic gradient descent implementation
    """
    def update_parameters(self, l, params, grads):
        # TODO: l input not needed here, but needed for ADAM - resolve nicely?
        """ Update the model parameters
        :param l: layer number
        :param params: model parameters
        :param grads: model parameter gradients
        :return: new_params: dictionary of updated parameters
        """
        new_params = dict()
        for p in params:
            new_params[p] = params[p] - self.alpha * grads['d'+p]

        return new_params


class ADAM(Base):
    """ ADAM (ADAptive Momentum) optimiser implementation
    """
    def __init__(self, alpha, beta_1=0.9, beta_2=0.999, eps=10**(-8)):
        """ Initialise the optimiser
        :param alpha: learning rate
        :param beta_1: momentum coefficient
        :param beta_2: RMSprop coefficient
        :param eps: eps: divide-by-0 protection
        """
        super().__init__(alpha)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.averages = dict()

    def compile(self, l, params):
        """ Initialise moving averages for the gradients in each layer and set
        iteration counter 't' to 0 (used for bias correction)
        :param l: layer number
        :param params: parameter dict keys
        """
        self.averages[l] = {'t': 0}
        self.averages[l].update({'V_d'+p: 0 for p in params})
        self.averages[l].update({'S_d'+p: 0 for p in params})

    def update_parameters(self, l, params, grads):
        """ Update the model parameters
        :param l: current layer
        :param params: model parameters
        :param grads: parameter gradients
        :return: new_params: dictionary of updated parameters
        """
        new_params = dict()
        self.averages[l]['t'] += 1
        for p in params.keys():
            V = 'V_d'+p
            S = 'S_d'+p
            self.averages[l][V] = self.beta_1*self.averages[l][V] + (1 - self.beta_1)*grads['d'+p]
            self.averages[l][S] = self.beta_2*self.averages[l][S] + (1 - self.beta_2)*grads['d'+p]**2
            V_corrected = self.averages[l][V] / (1 - self.beta_1**self.averages[l]['t'])
            S_corrected = self.averages[l][S] / (1 - self.beta_2**self.averages[l]['t'])
            new_params[p] = params[p] - (self.alpha * V_corrected / (S_corrected**0.5 + self.eps))

        return new_params
