from scipy.special import expit

class NeuralNet(object):
    """An implementation of a feed forward neural network."""

    def __init__(self, num_features, num_output=1, num_hidden=1, 
            activation="sigmoid"):
        """Constructor for the NeuralNet class.

        num_features:   The number of features that each sample has. This will
                        equal the number of neurons in the input layer.
        num_output:     The number of output labels each sample has. This will
                        equal the number of neurons in the output layer.
        num_hidden:     The number of hidden layers of the neural network.
        activation:     The default activation function to use in each neuron.
                        Default is the sigmoid function: s(x) = 1/(1 + e^(-x))
                        TODO: Create a list of the default functions.
        """

        # TODO: Address this.
        if num_output != 1:
            raise NotImplementedError("Neural network containing more than one
            output label has not been implemented yet.")

        # TODO: Address this.
        if num_hidden != 1:
            raise NotImplementedError("Neural network containing more than one
            hidden layer has not been implemented yet.")

        self.num_features = num_features
        self.num_output = num_output
        self.num_hidden = num_hidden

        # Assign activation function here, depending on the argument.
        if acivation == "sigmoid":
            self.default_act = expit
        else:
            raise NotImplementedError("Neural network that uses a default function
            other than the sigmoid function is not yet implemented.")

