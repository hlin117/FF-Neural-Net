from neuralnet import NeuralNet

class ClassifierNeuralNet(NeuralNet):
    """Acts as a wrapper around a normal neural network to provide an interface 
    to implement a classifier using a neural network.
    """

    def __init__(self, num_features, hidden_layer=None, activation="sigmoid", 
                num_classes=2, learn_rate=1, default_bias=1, stop_error=0.05,
                initial_scale=1):
        """Constructor for the RegressionNeuralNet class. The default number of
        output neurons is set to 1.

        See neuralnet/NeuralNet for constructor information.
        """
        if num_classes < 2:
            raise ValueError("Invalid number of classes to classify.")

        elif num_classes > 2:
            raise NotImplementedError("Classifying more than two classes has \
            not been implemented yet.")
        else:
            self.num_classes = 2
            self.threshold = 0.5

            super(self.__class__, self).__init__(num_features, 1,
                    hidden_layer, activation, learn_rate=learn_rate,
                    default_bias=default_bias, initial_scale=initial_scale)

    def set_threshold(self, new_threshold):
        """In the case of a binary classification problem, sets the threshold
        for which to consider a sample x as positive or negative.
        """
        if self.num_classes != 2:
            raise ValueError("Cannot set thresholds for neural network that \
            distinguishes more than two classes.")

        self.threshold = new_threshold
                        
    def score(self, sample):
        """Returns 1 if and only if score(sample) >= 0.

        Note that the true returned value is a matrix, so we have to
        do score(sample)[0, 0]
        """
        value = super(self.__class__, self).score(sample)[0, 0]
        return 1 if value >= self.threshold else -1
