from scipy.special import expit

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
            raise NotImplementedError("Neural network containing more than one
            output label has not been implemented yet.")

        # TODO: Address this.
        if len(hidden_layer) > 1
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

        # List of the weight matrices here. If w_ij(k) is a weight in the matrix,
        # it is the weight assigned to the edge from neuron i to neuron j in the
        # kth layer from the input. (With k = 1, we have the first hidden layer.)
        # TODO
        raise NotImplementedError("TODO")
