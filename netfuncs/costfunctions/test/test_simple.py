from netfuncs.costfunctions.simple import *
import numpy as np
import matplotlib.pyplot as plt


""" Test binary classifier cost """
print('TEST: simple.ClassifyBinary')
fn = BinaryCrossEntropy()
k = 0.01
ypred = np.arange(k, 1, k)
J = np.zeros((len(ypred), 2))
for i, yi in enumerate(ypred):
    J[i, 0] = fn.cost(np.array([[yi]]), np.array([[0]]))
    J[i, 1] = fn.cost(np.array([[yi]]), np.array([[1]]))
plt.figure()
plt.title('TEST: ClassifyBinary')
plt.plot(ypred, J[:, 0], label='Y={}'.format(0))
plt.plot(ypred, J[:, 1], label='Y={}'.format(1))
plt.xlabel('ypred')
plt.ylabel('J')
plt.legend()


""" Test multiclass classifier cost """
# TODO
print('TEST: simple.ClassifyMulti')
fn = CategoricalCrossEntropy()
k = 0.01
m = 5
# ypred = np.

plt.show()
