from scipy.special import expit
import numpy as np

class NeuralNet(object):
    """An implementation of a feed forward neural network."""

    def __init__(self, num_features, num_output=1, hidden_layer=[num_features*1.5], 
            activation="sigmoid"):
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
        else:
            raise NotImplementedError("Neural network that uses a default function \
                    other than the sigmoid function is not yet implemented.")

        # List of the weight matrices here. If w_ij(k) is a weight in the matrix,
        # it is the weight assigned to the edge from neuron i to neuron j in the
        # kth layer from the input. (With k = 1, we have the first hidden layer.)
        self.init_weights()

    def init_weights(self)
        """Initializes the weights on the edges between neurons.
        
        TODO: Extend this neural network to allow for more than one hidden layer.
        """
        self.weights1 = np.random.rand(self.num_features + 1, self.num_hidden)
        self.weights2 = np.random.rand(self.num_features + 1, self.num_hidden)
        self.weights1[-1] = 1
        self.weights2[-1] = 1

    def verity_data(self, data):
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
            outputs = self.feed_forward(sample) 
            errors = self.backpropagate(outputs, targets)
            update_weights(errors)

    def feed_forward(self, sample):
        """Obtains the output from a feedforward computation.
        
        NOTE: Still under development. The constructor of the neural
        network should prevent any NN with more than one hidden layer
        from being constructed."""
        input_aug = np.array(sample, dtype=float).append(1)
        excite1 = input_aug * self.weights1
        output1 = self.default_act(excite1)

        output1_aug = output1.append(1)
        excite2 = output1_aug * self.weights2
        output2 = self.default_act(excite2)

        return (output1, output2)

    def backpropagate(outputs, targets):
        pass
