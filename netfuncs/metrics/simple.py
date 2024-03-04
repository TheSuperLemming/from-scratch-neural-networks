class Accuracy:
    """ Measure model accuracy
    """

    def __init__(self):
        self.name = "accuracy"

    @staticmethod
    def evaluate(indices):
        """ Evaluate performance metric
        :param indices: dict of correctly and incorrectly predicted sample indices
        :return accuracy: model accuracy
        """
        n_correct = len(indices['correct'])
        n_incorrect = len(indices['incorrect'])
        accuracy = n_correct / (n_correct + n_incorrect)

        return accuracy

