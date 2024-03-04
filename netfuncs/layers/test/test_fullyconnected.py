from netfuncs.layers.fullyconnected import FC
from netfuncs.activations.simple import *
from netfuncs.optimisers.simple import GradDescent
from netfuncs.costfunctions.simple import *
import matplotlib.pyplot as plt
import numpy as np
import copy


def check_gradient(L, X, Y, costfunction):
    """ Numerical calculation of gradients
    :param L: layer object
    :param X: input data
    :param Y: labels array
    :return dtheta: numerical gradients
    """
    flatW = L.W.flatten()
    flatb = L.b.flatten()
    nW = len(flatW)
    nb = len(flatb)
    theta = np.concatenate((flatW, flatb))
    dtheta = np.zeros(theta.shape)
    eps = 10**(-7)

    for i in range(len(theta)):
        theta_plus = copy.deepcopy(theta)
        theta_minus = copy.deepcopy(theta)
        theta_plus[i] += eps
        theta_minus[i] -= eps

        L.W = np.reshape(theta_plus[:nW], L.W.shape)
        L.b = np.reshape(theta_plus[nW:], L.b.shape)
        L.forward_prop(X)
        J_plus = costfunction.cost(L.A, Y)

        L.W = np.reshape(theta_minus[:nW], L.W.shape)
        L.b = np.reshape(theta_minus[nW:], L.b.shape)
        L.forward_prop(X)
        J_minus = costfunction.cost(L.A, Y)

        dtheta[i] = (J_plus - J_minus) / (2*eps)

    return dtheta


""" Set constants """
size = 5  # number of neurons in layer
input_size = 10  # number of elements in feature vector
m = 20  # number of test samples


""" Initalise layer """
if size == 1:
    activation = Sigmoid()
    costfunction = BinaryCrossEntropy()
else:
    activation = Softmax()
    costfunction = CategoricalCrossEntropy()
optimiser = GradDescent(alpha=0.1)
layer = FC(size=size, activation=activation, optimiser=optimiser)


""" Test weights/bias initialisation """
print('TEST: layer.initialise')
layer.initialise(input_size)
passW = layer.W.shape == (size, input_size)
passb = layer.b.shape == (size, 1)
print('> check W >> test: expected shape={}, got {} >> test passed={}'.format((size, input_size), layer.W.shape, passW))
print('> check b >> test: expected shape={}, got {} >> test passed={}'.format((size, 1), layer.b.shape, passb))


""" Test forward propagation """
print('TEST: layer.forward_prop')
X = np.random.randn(input_size, m)
layer.forward_prop(X)
passZ = (layer.Z == (layer.W@X + layer.b)).all()
print('> check Z >> test: Z=WX+b >> test passed={}'.format(passZ))
if type(activation) == Softmax:
    eps = 10 ** (-7)
    passA = (np.sum(layer.A, axis=0)-np.ones((1, m)) < eps).all()
    print('> check A >> test: all A(i) sum to one >> test passed={}'.format(passA))
else:
    print('> check A >> verify distribution from plot')
    testZ = np.arange(-10, 10, 0.1)
    plt.figure()
    plt.title('TEST: layer.forward_prop (activations)')
    plt.plot(testZ, activation.f(testZ), label='expected distribution')
    plt.scatter(layer.Z, layer.A, c='r', label='layer output')
    plt.legend()


""" Test back propagation """
print('TEST: layer.back_prop')
eps = 10**(-7)
testY = np.zeros(layer.A.shape)
testY[0, :] = 1
testdtheta = check_gradient(copy.deepcopy(layer), X, testY, costfunction)
layer.back_prop(X, dZ=costfunction.dZ(layer.A, testY))
dtheta = np.concatenate((layer.gd['dW'].flatten(), layer.gd['db'].flatten()))
delta = np.linalg.norm(testdtheta-dtheta) / (np.linalg.norm(testdtheta) + np.linalg.norm(dtheta))
passgrads = delta < eps
print('> check grads >> test: threshold eps={} >> test passed={}'.format(eps, passgrads))

plt.show()
