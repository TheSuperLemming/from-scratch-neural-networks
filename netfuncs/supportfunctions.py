import numpy as np
import math
import copy

import matplotlib.pyplot as plt


def true_round(x):
    """ Implements round "away from zero" in lieu of python's default "ties to even"
    :param x: number or sequence of numbers to round
    :return round_x: numpy array of rounded values
    """
    x = np.array(x)
    round_x = np.zeros(x.shape)
    for i, xi in enumerate(x):
        frac = xi - math.floor(xi)
        round_x[i] = math.floor(xi) + (frac >= 0.5)

    return round_x.astype(int)


def check_gradient(model, X, Y, parameters=("W", "b")):
    """ Numerical calculation of gradients
    :param model: model object
    :param X: input data
    :param Y: labels array
    :param parameters: model parameters to check
    :return dtheta: numerical gradients
    """
    # Vectorise all model parameters
    model_vec, model_map = model_to_vector(model, parameters)
    N = len(model_vec)

    # Train model for 1 epoch to calculate grads
    model.train(X, Y, num_epochs=1)
    grad_vec, grad_map = grads_to_vector(model, parameters)

    # Compute grads numerically
    numerical_grads = np.zeros(model_vec.shape)
    eps = 10 ** (-9)  # perturbation
    for i in range(N):
        if (i + 1) % 100 == 0:
            print("numerical grad check completed: {} of {}".format(i + 1, N))

        # Perturb each parameter in turn
        vec_plus = copy.deepcopy(model_vec)
        vec_minus = copy.deepcopy(model_vec)
        vec_plus[i] += eps
        vec_minus[i] -= eps

        # Compute J_plus
        model = vector_to_model(model, vec_plus, model_map)
        _ = model.predict(X)
        J_plus = model.cost_fun.cost(model.layers[-1].A, Y)

        # Compute J_minus
        model = vector_to_model(model, vec_minus, model_map)
        _ = model.predict(X)
        J_minus = model.cost_fun.cost(model.layers[-1].A, Y)

        numerical_grads[i] = (J_plus - J_minus) / (2 * eps)

    check = np.linalg.norm(numerical_grads - grad_vec) / (np.linalg.norm(numerical_grads) + np.linalg.norm(grad_vec))
    is_good = check < eps

    print("is_good: {}".format(is_good))

    fig = plt.figure("grad_check")
    plt.plot(numerical_grads, label='num')
    plt.plot(grad_vec, marker='o', markersize=5, linestyle='', label='gd')
    ybar = 1.5 * np.max(abs(numerical_grads))
    for l in grad_map:
        for param in grad_map[l]:
            pointers = grad_map[l][param]["pointers"]
            plt.plot([pointers[0], pointers[0]], [-ybar, ybar], color='k')
            plt.plot([pointers[1], pointers[1]], [-ybar, ybar], color='k')
            plt.annotate(param+str(l), (pointers[0], ybar))
    plt.legend()

    fig = plt.figure("grad_check_deltas")
    plt.plot(grad_vec - numerical_grads)
    ybar = 1.5 * np.max(abs(grad_vec - numerical_grads))
    for l in grad_map:
        for param in grad_map[l]:
            pointers = grad_map[l][param]["pointers"]
            plt.plot([pointers[0], pointers[0]], [-ybar, ybar], color='k')
            plt.plot([pointers[1], pointers[1]], [-ybar, ybar], color='k')
            plt.annotate(param + str(l), (pointers[0], ybar))

    plt.show()

    return


def model_to_vector(model, parameters):
    """ Vectorise all model parameters in the "parameters" list
    :param model: model object
    :param parameters: model parameters to check
    :return model_vec: vector of model parameters
    :return model_map: dictionary containing information to restore model from model_vec
    """
    model_vec = np.array([])
    model_map = {}
    for l in range(model.num_layers):
        model_map[l] = {}
        for param in parameters:
            model_map[l][param] = {"shape": getattr(model.layers[l], param).shape,
                                   "pointers": len(model_vec) + np.array([0, getattr(model.layers[l], param).size])}
            model_vec = np.concatenate((model_vec, getattr(model.layers[l], param).flatten()))

    return model_vec, model_map


def grads_to_vector(model, parameters):
    """ Vectorise all gradient parameters in the "parameters" list
    :param model: model object
    :param parameters: model parameters to check
    :return grad_vec: vector of gradient parameters
    :return grad_map: dictionary containing information to restore model from grad_vec
    """
    grad_vec = np.array([])
    grad_map = {}
    for l in range(model.num_layers):
        grad_map[l] = {}
        for p in parameters:
            param = "d" + p
            grad_map[l][param] = {"shape": model.layers[l].gd[param].shape,
                                  "pointers": len(grad_vec) + np.array([0, model.layers[l].gd[param].size])}
            grad_vec = np.concatenate((grad_vec, model.layers[l].gd[param].flatten()))

    return grad_vec, grad_map


def vector_to_model(model, model_vec, model_map):
    """ Restore model from model_vec
    :param model: model object
    :param model_vec: vector of model parameters
    :param model_map: dictionary containing information to restore model from model_vec
    :return model: restored model object
    """
    for l in model_map.keys():
        for param in model_map[l].keys():
            pointers = model_map[l][param]["pointers"]
            shape = model_map[l][param]["shape"]
            value = model_vec[pointers[0]:pointers[1]].reshape(shape)
            setattr(model.layers[l], param, value)

    return model
