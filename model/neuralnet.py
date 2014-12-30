from __future__ import print_function
import json
from scipy.special import expit
import math
import numpy as np

one_line = lambda string: string.replace("\n", "").replace("\t", "")

class NeuralNet(object):
    """An implementation of a feed forward neural network."""

    def __init__(self, num_features, num_output=1, hidden_layer=None, 
                 activation="expit", learn_rate=1, default_bias=0, max_epochs=10,
                 scale=1, verbose=False):
        """Constructor for the NeuralNet class.

        num_features:   The number of features that each sample has. This will
                        equal the number of neurons in the input layer.
        num_output:     The number of output labels each sample has. This will
                        equal the number of neurons in the output layer.
        hidden_layer:   A list containing the number of nodes in the (i+1)th 
                        hidden layer (for i starting at 0). If set to None (default
                        value), then the neural network will have one hidden
                        layer with num_features * 1.5 hidden nodes.
        activation:     The default activation function to use in each neuron.
                        Default is the inverse logistic function: 
                        s(x) = 1/(1 + e^(-x)). This uses scipy for optimization
                        purposes.
        learn_rate:     The learning rate applied to the training process. Default
                        value is 1.
        default_bias:   The default weight assigned to the weight vector. Default
                        value is 0.
        max_epochs:     The max number of iterations on the training set. Default
                        value is 10.
        scale:          Determines the range of random values for the initial
                        weights of the model. The value of the weights will range
                        from (-scale, scale). For example, if scale=2, then the
                        initial weights can range from (-scale, scale). Default
                        value is 1.
        verbose:        Used to see how fast the neural network is being trained.
                        Indicates when an epoch has finished.
        """

        self.verbose = verbose
        self.num_features = num_features
        self.num_output = num_output
        self.hidden_layer = hidden_layer

        self.learn_rate = learn_rate
        self.default_bias = default_bias
        self.error = lambda x, y: 0.5 * (x - y)**2  # Least means square

        self.max_epochs = max_epochs
        self.scale = scale

        # NOTE: There is no proven evidence that the ideal number of
        # nodes in the hidden layer is 1.5, but it is suggested. Citation needed.
        if self.hidden_layer is None:
            num_nodes = int(math.floor(num_features * 1.5))
            self.hidden_layer = [num_nodes]

        # TODO: Address this.
        if len(self.hidden_layer) > 1:
            raise NotImplementedError(one_line("""Neural network containing 
            more than one hidden layer has not been implemented yet."""))

        # TODO: Address this.
        if self.num_output != 1:
            raise NotImplementedError(one_line("""Neural network containing more 
            than one output label has not been implemented yet."""))

        # Assign activation function here, depending on the argument.
        if activation == "expit":
            self.default_act = expit
            self.default_deriv = lambda x: expit(x) * (1 - expit(x))
        else:
            raise NotImplementedError(one_line("""Neural network that uses a 
            default function other than the sigmoid function is not yet 
            implemented."""))

        self.init_weights()

    @classmethod
    def load(cls, filename):
        """A method to create a neural network by loading a file in json-format"""
        json_obj = json.load(open(filename))

        # len(json_obj) + 1 denotes the number of layers in the neural network.
        if len(json_obj) + 1 != 3:
            raise NotImplementedError(one_line("""Currently only supporting 
            neural network with only one hidden layer."""))

        weights1 = np.mat(json_obj[0]).T
        weights2 = np.mat(json_obj[1]).T

        nn = cls(weights1.shape[0] - 1, hidden_layer=[weights1.shape[1]])
        nn.weights1 = weights1
        nn.weights2 = weights2
        return nn
        

    def init_weights(self):
        """Initializes the weights on the edges between neurons.
        Weights are initialized to a random float between (-scale, scale).
        
        TODO: Extend this neural network to allow for more than one
        hidden layer.
        """
        rescale = lambda matrix: self.scale * matrix - self.scale / 2
        self.weights1 = np.mat(rescale(np.random.rand(self.num_features + 1,
                               self.hidden_layer[0])))
        self.weights2 = np.mat(rescale(np.random.rand(self.hidden_layer[0] + 1,
                               self.num_output)))
        self.weights1[-1, :] = self.default_bias
        self.weights2[-1, :] = self.default_bias


    def verify_data(self, data):
        """Verifies that the data is in the form of a nested iterable, and 
        that each is of the length self.num_features. Also verifies that
        each inner nested object is a float type object.

        Might want to introduce a command that disables this feature, but
        as far as I know, only the training process is known to take much time.
        """
        for sample in data:
            if len(sample) != self.num_features:
                message = "Input data is not of the same length \
                        as the number of input neurons. Received {0}, not {1} \
                        ".replace("\n", "").replace("  ", "") \
                        .format(len(sample), self.num_features)
                raise ValueError(message)

            for feature in sample:
                if not isinstance(feature, (int, float)):
                    raise ValueError("Detected feature that is not compatible \
                            with the Neural Network: {0}".replace("\n", "").format(feature))

    def train(self, data, targets):
        """Trains the neural network on a set of data. Data should be
        in the form of two nested ordered iterables. Uses the backpropagation
        algorithm to adjust the weights of the edges.

        Each inner iterable should be of length num_features. If not, then
        a ValueError is raised.

        Continues training until we have pass through the training set
        at least one, and the error is sufficiently small."""
        self.verify_data(data)

        for i in xrange(self.max_epochs):
            self.verbose_print("Starting epoch {0}".format(i + 1))
            for j, sample in enumerate(data):
                outputs = self.feed_forward(sample, all_layers=True)
                deltas = self.backpropagate(outputs, targets[j])
                self.update_weights(deltas, outputs)

                # Check whether we should break out
#                error = self.error(outputs[-1][0, 0], targets[i][0])
#                if error < self.stop_error and not first_pass:
#                    large_error = False
#                    break
    
    def verbose_print(self, string):
        if self.verbose: print(string)

    def feed_forward(self, sample, all_layers=False):
        """Obtains the output from a feedforward computation.
        
        TODO: Still under development. The constructor of the neural
        network should prevent any NN with more than one hidden layer
        from being constructed."""
        # Represents the augmented input data
        input_aug = np.mat(np.append(np.array(sample, dtype=float), 1))

        # Calculates the augmented output of the hidden layer
        excite1 = input_aug * self.weights1
        output1 = self.default_act(excite1)
        output1_aug = np.mat(np.append(np.array(output1), 1))

        # Calculates the (non-augmented) output of the output layer
        excite2 = output1_aug * self.weights2
        output2 = self.default_act(excite2)

        if all_layers:
            return input_aug, output1_aug, output2
        else:
            return output2

    def backpropagate(self, outputs, targets):
        """Performs the backpropogation algorithm to determine the error.

        TODO: The current implementation assumes that
        1. The error function to minimize is least-squares
        2. The activation function used is the sigmoid
        3. There is only one hidden layer

        I'm going to need to update this code later on in the future...

        TODO: Explain the naming convention of variables with more detail
        in the future.
        """

        if len(outputs) != 3:
            raise ValueError("Currently only expecting three output vectors \
                    from each of the layers.")

        if len(outputs[-1]) != 1 or len(targets) != 1:
            raise ValueError("Current implementation cannot handle \
                    more than one output neuron.")

        # NOTE: Assuming that the output has already been calculated
        # x is an output of the sigmoid function.
        sig_deriv = np.vectorize(lambda x: x * (1 - x))
        derivs2 = np.diag(sig_deriv(outputs[2]))

        # NOTE: The "diag" command only works with nd arrays that are flattened...
        derivs1 = np.diag(np.asarray(sig_deriv(outputs[1][:, :-1])).flatten())

        error_deriv = np.mat(np.array(outputs[-1] - targets))
        delta2 = derivs2 * error_deriv
        delta1 = derivs1 * self.weights2[:-1, :] * delta2

        # delta2 and delta1 will be the "correction" that we have to
        # apply to weights2 and weights1
        return delta1, delta2

    def update_weights(self, deltas, outputs):
        """Updates the weights of the edges."""
        self.weights2 += (-self.learn_rate * deltas[1] * outputs[1]).T
        self.weights1 += (-self.learn_rate * deltas[0] * outputs[0]).T

    def score_data(self, data):
        """Performs predictions for each of the values stored in data.
        Returns a p-length tuple of predictions for each of the p samples.
        """
        self.verify_data(data)
        return tuple(self.score(sample) for sample in data)

    def classify(self, sample, threshold):
        """Performs binary classification. TODO: Add option for k-class
        classification """
        value = self.score(sample)
        return 1 if value >= threshold else -1

    def classify_data(self, data, threshold):
        """Classifies each of the samples. Returns a tuple representing
        the results of the classification task."""
        return tuple(self.classify(sample, threshold) for sample in data)

    def score(self, sample):
        return self.feed_forward(sample)


