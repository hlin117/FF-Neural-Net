from scipy.special import expit
import numpy as np

class NeuralNet(object):
    """An implementation of a feed forward neural network."""

    def __init__(self, num_features, num_output=1, hidden_layer=[num_features*1.5], 
            activation="sigmoid", learn_rate=1):
        """Constructor for the NeuralNet class.

        num_features:   The number of features that each sample has. This will
                        equal the number of neurons in the input layer.
        num_output:     The number of output labels each sample has. This will
                        equal the number of neurons in the output layer.
        hidden_layer:   A list containing the number of nodes in the (i+1)th 
                        hidden layer (for i starting at 0).
        activation:     The default activation function to use in each neuron.
                        Default is the sigmoid function: s(x) = 1/(1 + e^(-x))
                        TODO: Create a list of the default functions.
        """

        # TODO: Address this.
        if num_output != 1:
            raise NotImplementedError("Neural network containing more than one \
                    output label has not been implemented yet.")

        # TODO: Address this.
        if len(hidden_layer) > 1
            raise NotImplementedError("Neural network containing more than one \
                    hidden layer has not been implemented yet.")

        self.num_features = num_features
        self.num_output = num_output
        self.num_hidden = num_hidden

        # Assign activation function here, depending on the argument.
        if acivation == "sigmoid":
            self.default_act = expit
            self.default_deriv = lambda x : expit(x) * (1 - expit(x))
        else:
            raise NotImplementedError("Neural network that uses a default function \
                    other than the sigmoid function is not yet implemented.")

        # List of the weight matrices here. If w_ij(k) is a weight in the matrix,
        # it is the weight assigned to the edge from neuron i to neuron j in the
        # kth layer from the input. (With k = 1, we have the first hidden layer.)
        self.learn_rate = learn_rate
        self.init_weights()

    def init_weights(self)
        """Initializes the weights on the edges between neurons.
		Weights are initialized to random values between -1 and 1.
        
        TODO: Extend this neural network to allow for more than one 
        hidden layer.
        """
		rescale = lambda matrix : 2 * matrix - 1
        self.weights1 = rescale(np.random.rand(self.num_features + 1, 
                self.num_hidden)
        self.weights2 = rescale(np.random.rand(self.num_features + 1, 
                self.num_hidden))
        self.weights1[-1] = 1
        self.weights2[-1] = 1

    def verify_data(self, data):
        """Verifies that the data is in the form of a nested iterable, and 
        that each is of the length self.num_features. Also verifies that
        each inner nested object is a float type object.

        Might want to introduce a command that disables this feature, but
        as far as I know, only the training process is known to take much time.
        """
        for sample in data:
            if len(sample) != self.num_features:
                raise ValueError("Input data is not of the same length \
                        as the number of input neurons: {0}".format(innerlist))

            for feature in sample:
                if isinstance(feature, (int, float)):
                    raise ValueError("Detected feature that is not compatible \
                            with the Neural Network: {0}".format(feature))

    def train(self, data, targets)
        """Trains the neural network on a set of data. Data should be
        in the form of two nested ordered iterables. Uses the backpropagation
        algorithm to adjust the weights of the edges.

        Each inner iterable should be of length num_features. If not, then
        a ValueError is raised."""
        self.verify_data(data)
        for sample in data:
            outputs = self.feed_forward(sample, all_layers=True) 
            deltas = self.backpropagate(outputs, targets)
            self.update_weights(deltas, outputs)

    def feed_forward(self, sample, all_layers=False):
        """Obtains the output from a feedforward computation.
        
        NOTE: Still under development. The constructor of the neural
        network should prevent any NN with more than one hidden layer
        from being constructed."""
        # Represents the augmented input data
        input_aug = np.array(sample, dtype=float).append(1)

        # Calculates the augmented output of the hidden layer
        excite1 = input_aug * self.weights1
        output1 = self.default_act(excite1)
        output1_aug = output1.append(1)

        # Calculates the (non-augmented) output of the output layer
        excite2 = output1_aug * self.weights2
        output2 = self.default_act(excite2)

        return input_aug, output1_aug, output2

    def backpropagate(outputs, targets):
        """Performs the backpropogation algorithm to determine the error.

        TODO: The current implementation assumes that
        1. The error function to minimize is least-squares
        2. The activation function used is the sigmoid
        3. There is only one hidden layer

        I'm going to need to update this code later on in the future...

        TODO: Explain the naming convention of variables with more detail
        in the future.
        """

        if len(outputs != 3):
            raise ValueError("Currently only expecting three output vectors \
                    for each of the layers.")

        if len(outputs[-1]) != 1) or len(targets[-1] != 1):
            raise ValueError("Current implementation cannot handle \
                    more than one output neuron.")

        # NOTE: Assuming that the output has already been calculated
        # x is an output of the sigmoid function.
        sig_deriv = lambda x : x * (1 - x)
        derivs2 = np.diag(sig_deriv(outputs[2]))
        derivs1 = np.diag(sig_deriv(outputs[1]))

        error_deriv = np.array(outputs[-1] - targets)
        delta2 = derivs2 * error_deriv
        delta1 = derivs1 * self.weights2[:-1] * delta2

        # delta2 and delta1 will be the "correction" that we have to
        # apply to weights2 and weights1
        return delta1, delta2

    def update_weights(self, deltas, outputs):
        """Updates the weights of the edges."""
        self.weights2.T += -self.learn_rate * deltas[1] * outputs[1]
        self.weights1.T += -self.learn_rate * deltas[0] * outputs[0]
	
    def score_data(self, data):
            """Performs predictions for each of the values stored in data.
            
            Returns a p-length tuple of predictions for each of the p samples.
            """
            self.verify_data(data)
            return tuple(self.score(sample) for sample in data)
            
    def score(self, sample):
            return self.feed_forward(sample)
