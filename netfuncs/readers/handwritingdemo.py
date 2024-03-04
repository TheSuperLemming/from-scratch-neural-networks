from netfuncs.supportfunctions import true_round
import numpy as np
import csv
import matplotlib.pyplot as plt


class ReadHandwritingFC:
    """ Reader class to parse input data from the example csv files.
    Samples are 20x20 pixel arrays saved as 400x1 vectors.
    Samples are labelled as numbers 0-9 (0 represented as 10)
    """

    def read(self, file_path, split):
        """ Read data X and labels Y
        :param file_path: string containing path to test files
        :param split: list containing train/dev/test set ratios
        :return X: vectorised image samples (test/dev/train datasets)
        :return Y: vectorised label data (test/dev/train datasets)
        """
        if len(split) != 3:
            raise(Exception("need 3 values in split for train/dev/test ratios"))
        if sum(split) != 1:
            raise(Exception("split values must sum to 1"))

        X = []  # Initialise input data array
        Y = []  # Initialise truth data array

        with open(file_path + "X.csv") as file:
            read_X = csv.reader(file)
            for row in read_X:
                X.append(row)

        with open(file_path + "Y.csv") as file:
            read_Y = csv.reader(file)
            for row in read_Y:
                Y.append(row)

        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=int)
        m = X.shape[0]

        # Re-parse Y from class id to vector of 0s and 1s per class
        Y_vec = self.convert_label_to_vector(Y)

        # Randomly shuffle data
        new_order = np.random.permutation(m)
        X = X[new_order, :].T
        Y = Y_vec[new_order, :].T

        count = true_round([m*n for n in split])

        X_train = X[:, np.arange(count[0])]
        Y_train = Y[:, np.arange(count[0])]
        X_dev = X[:, np.arange(count[1]) + count[0]]
        Y_dev = Y[:, np.arange(count[1]) + count[0]]
        X_test = X[:, np.arange(count[2]) + sum(count[:2])]
        Y_test = Y[:, np.arange(count[2]) + sum(count[:2])]

        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

    @staticmethod
    def convert_label_to_vector(Y):
        """ Convert single-valued class labels to one-hot vector representation
        :param Y: single-value class labels
        :return Y_vec: one-hot vector representation
        """
        Y_vec = np.zeros([Y.shape[0], Y.max()])
        for n, class_id in enumerate(Y):
            Y_vec[n, class_id - 1] = 1

        return Y_vec

    @staticmethod
    def convert_vector_to_label(Y_vec):
        """ Convert one-hot representation to single-value class labels
        :param Y_vec: one-hot vector representation
        :return Y: single-value class labels
        """
        Y = np.argmax(Y_vec) + 1

        return Y

    def plot_sample(self, X, Y, n):
        """ Plot sample image and label
        :param X: sample data
        :param Y: label data
        :param n: number of samples to plot
        """
        side = np.ceil(n**0.5).astype('int')
        _, m = X.shape
        i_sample = np.random.randint(0, m, size=n)

        plt.figure("plot samples")
        for p, i in enumerate(i_sample):
            plt.subplot(side, side, p+1)
            plt.imshow(X[:, i].reshape(20, 20).T)
            plt.title("y={}".format(self.convert_vector_to_label(Y[:, i])))
            plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)


class ReadHandwritingCONV:
    """ Reader class to parse input data from the example csv files.
    Samples are 20x20 pixel - by - 1 channel arrays saved as 400x1 vectors.
    Samples are labelled as numbers 0-9 (0 represented as 10)
    """

    def read(self, file_path, split):
        """ Read data X and labels Y
        :param file_path: string containing path to test files
        :param split: list containing train/dev/test set ratios
        :return X: 1-channel image samples (test/dev/train datasets)
        :return Y: vectorised label data (test/dev/train datasets)
        """
        if len(split) != 3:
            raise(Exception("need 3 values in split for train/dev/test ratios"))
        if sum(split) != 1:
            raise(Exception("split values must sum to 1"))

        X = []  # Initialise input data array
        Y = []  # Initialise truth data array

        with open(file_path + "X.csv") as file:
            read_X = csv.reader(file)
            for row in read_X:
                X.append(row)

        with open(file_path + "Y.csv") as file:
            read_Y = csv.reader(file)
            for row in read_Y:
                Y.append(row)

        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=int)
        m, _ = X.shape

        # Re-parse Y from class id to vector of 0s and 1s per class
        Y_vec = self.convert_label_to_vector(Y)

        # Randomly shuffle data
        new_order = np.random.permutation(m)
        X = np.reshape(X[new_order, :].T, (20, 20, 1, m))
        Y = Y_vec[new_order, :].T

        count = true_round([m*n for n in split])

        X_train = X[:, :, :, np.arange(count[0])]
        Y_train = Y[:, np.arange(count[0])]
        X_dev = X[:, :, :, np.arange(count[1]) + count[0]]
        Y_dev = Y[:, np.arange(count[1]) + count[0]]
        X_test = X[:, :, :, np.arange(count[2]) + sum(count[:2])]
        Y_test = Y[:, np.arange(count[2]) + sum(count[:2])]

        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

    @staticmethod
    def convert_label_to_vector(Y):
        """ Convert single-valued class labels to one-hot vector representation
        :param Y: single-value class labels
        :return Y_vec: one-hot vector representation
        """
        Y_vec = np.zeros([Y.shape[0], Y.max()])
        for n, class_id in enumerate(Y):
            Y_vec[n, class_id - 1] = 1

        return Y_vec

    @staticmethod
    def convert_vector_to_label(Y_vec):
        """ Convert one-hot representation to single-value class labels
        :param Y_vec: one-hot vector representation
        :return Y: single-value class labels
        """
        Y = np.argmax(Y_vec) + 1

        return Y

    def plot_sample(self, X, Y, n):
        """ Plot sample image and label
        :param X: sample data
        :param Y: label data
        :param n: number of samples to plot
        """
        side = np.ceil(n**0.5).astype('int')
        m = X.shape[3]
        i_sample = np.random.randint(0, m, size=n)

        plt.figure("plot samples")
        for p, i in enumerate(i_sample):
            plt.subplot(side, side, p+1)
            plt.imshow(np.squeeze(X[:, :, :, i]).T)
            plt.title("y={}".format(self.convert_vector_to_label(Y[:, i])))
            plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
