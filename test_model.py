from netfuncs.readers.handwritingdemo import ReadHandwritingFC
from netfuncs.layers.fullyconnected import FC
from netfuncs.activations.simple import ReLU, Softmax
from netfuncs.optimisers.simple import ADAM
from netfuncs.costfunctions.simple import CategoricalCrossEntropy
from netfuncs.metrics.simple import Accuracy
from netfuncs.model import Model

''' Import data '''

file_path = "C:/Users/jason/OneDrive/Documents/Python/NewNeuralNet/data/handwritingdemo/"
split = [0.8, 0.1, 0.1]

reader = ReadHandwritingFC()
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = reader.read(file_path, split)

input_size = X_train.shape[0]
output_size = Y_train.shape[0]

# reader.plot_sample(X_train, Y_train, 16)

''' Build model '''

relu = ReLU()
softmax = Softmax()
cost_function = CategoricalCrossEntropy()
optimiser = ADAM(alpha=0.005)
# optimiser = GradDescent(alpha=0.3)
model_metrics = [Accuracy()]

layer1 = FC(25, relu)
layer2 = FC(output_size, softmax)
layers = [layer1, layer2]

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
