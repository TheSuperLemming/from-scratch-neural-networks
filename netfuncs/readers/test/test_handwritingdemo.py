from netfuncs.readers.handwritingdemo import ReadHandwritingFC
from netfuncs.supportfunctions import true_round
import random
import numpy as np
import matplotlib.pyplot as plt


file_path = "C:/Users/jason/OneDrive/Documents/Python/NewNeuralNet/data/handwritingdemo/"
split = [0.8, 0.1, 0.1]

m = 5000  # number of samples in handwritingdemo dataset
n = 400   # feature vector length
c = 10    # number of classes

reader = ReadHandwritingFC()
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = reader.read(file_path, split)

print('TEST: reader.read')
m_set = true_round([s*m for s in split])
passXtrain = (n, m_set[0]) == X_train.shape
passXdev = (n, m_set[1]) == X_dev.shape
passXtest = (n, m_set[2]) == X_test.shape
passYtrain = (c, m_set[0]) == Y_train.shape
passYdev = (c, m_set[1]) == Y_dev.shape
passYtest = (c, m_set[2]) == Y_test.shape
print('> check dataset sizes ')
print(' >> test X_train: expected shape={}, got {} >> test passed={}'.format((n, m_set[0]), X_train.shape, passXtrain))
print(' >> test X_dev: expected shape={}, got {} >> test passed={}'.format((n, m_set[1]), X_dev.shape, passXdev))
print(' >> test X_test: expected shape={}, got {} >> test passed={}'.format((n, m_set[2]), X_test.shape, passXtest))
print(' >> test Y_train: expected shape={}, got {} >> test passed={}'.format((c, m_set[0]), X_train.shape, passYtrain))
print(' >> test Y_dev: expected shape={}, got {} >> test passed={}'.format((c, m_set[1]), X_dev.shape, passYdev))
print(' >> test Y_test: expected shape={}, got {} >> test passed={}'.format((c, m_set[2]), X_test.shape, passYtest))
print('> check image output >> verify from plot')
i = [random.randint(0, m_set[0]), random.randint(0, m_set[1]), random.randint(0, m_set[2])]
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(X_train[:, i[0]].reshape(20, 20).T)
plt.gca().set_title("train #{} (y={})".format(i[0], np.argmax(Y_train[:, i[0]])+1))
plt.subplot(1, 3, 2)
plt.imshow(X_dev[:, i[1]].reshape(20, 20).T)
plt.gca().set_title("dev #{} (y={})".format(i[1], np.argmax(Y_dev[:, i[1]])+1))
plt.subplot(1, 3, 3)
plt.imshow(X_test[:, i[2]].reshape(20, 20).T)
plt.gca().set_title("test #{} (y={})".format(i[2], np.argmax(Y_test[:, i[2]])+1))

print('TEST: reader.plot')
i = random.randint(0, m_set[0])
reader.plot_sample(X_train[:, i], Y_train[:, i], 1)
print('> check sample >> verify from plot')

plt.show()
