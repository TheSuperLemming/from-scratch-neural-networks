from handwritingdemo import ReadHandwritingCONV
from layers.convolutional import CONV, FLATTEN
from layers.fullyconnected import FC
from activations.simple import ReLU, Softmax
from optimisers.simple import ADAM
from costfunctions.simple import CategoricalCrossEntropy
from metrics.simple import Accuracy
from model import Model

''' Import data '''

file_path = "C:/Users/jason/OneDrive/Documents/Python/NewNeuralNet/data/handwritingdemo/"
split = [0.8, 0.1, 0.1]

reader = ReadHandwritingCONV()
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = reader.read(file_path, split)

height, width, channels, _ = X_train.shape
input_size = (height, width, channels)
output_size = Y_train.shape[0]

reader.plot_sample(X_train, Y_train, 16)

''' Build model '''

relu = ReLU()
softmax = Softmax()
cost_function = CategoricalCrossEntropy()
optimiser = ADAM(alpha=0.005)
model_metrics = [Accuracy()]

layer1 = CONV(kernels=5, filter_size=3, activation=relu, pad=1)
layer2 = FLATTEN()
layer3 = FC(size=output_size, activation=softmax)
layers = [layer1, layer2, layer3]

model = Model(layers, cost_function, optimiser, metrics=model_metrics)

''' Compile and train model '''

model.compile(input_size)
model.train(X_train, Y_train, num_epochs=200, plot_cost=True)

''' Evaluate against training set '''

Y_pred = model.predict(X_train)
indices, metrics = model.evaluate(Y_pred, Y_train)

print('Training Set')
for m in metrics:
    print('> {}: {}'.format(m, metrics[m]))

''' Evaluate against test set '''

Y_pred = model.predict(X_test)
indices, metrics = model.evaluate(Y_pred, Y_test)

print('Test Set')
for m in metrics:
    print('> {}: {}'.format(m, metrics[m]))

''' Plot examples '''

plot_examples = True
if plot_examples:
    reader.plot_sample(X_test, Y_pred, 16)

print('Done')
