from neuralnet import NeuralNet

class ClassifierNeuralNet(object):
    """Acts as a wrapper around a normal neural network to provide an interface 
    to implement a classifier using a neural network.
    """

    def __init__(self, num_features, num_hidden=1, activation="sigmoid",
            num_classes=2):
        """Constructor for the RegressionNeuralNet class. The default number of
        output neurons is set to 1.

        num_features:   The number of features that each sample has. This will
                        equal the number of neurons in the input layer.
        num_hidden:     The number of hidden layers of the neural network.
        activation:     The default activation function to use in each neuron.
                        Default is the sigmoid function: s(x) = 1/(1 + e^(-x))
                        TODO: Create a list of the default functions.
        num_classes:    The number of classes to determine. Default is 2 for a
                        binary classification problem, in which case the
                        learning threshold for which to consider a positive or
                        negative example is 0.5. That is, a sample x is positive
                        if and only if score(x) >= t, where score() is the
                        assigned value to x by the neural network.

                        If num_classes > 2, then we will use the softmax function
                        to determine the class of x. Otherwise, the neural
                        network will contain only one neuron in the output layer.

                        TODO: May want to consider creating another class for
                        binary classifiers.
        """
        if num_classes < 2:
            raise ValueError("Invalid number of classes to classify.")

        elif num_classes > 2:
            raise NotImplementedError("Classifying more than two classes has
            not been implemented yet.")
        else:
            self.num_classes = 2
            self.threshold = 0.5

            super(self.__class__, self).__init__(num_features, 1)

    def set_threshold(self, new_threshold):
        """In the case of a binary classification problem, sets the threshold
        for which to consider a sample x as positive or negative.
        """
        if self.num_classes != 2:
            raise ValueError("Cannot set thresholds for neural network that
            distinguishes more than two classes.")

        self.threshold = new_threshold
                        
