#!/usr/bin/python
import numpy as np
from classneuralnet import ClassifierNeuralNet


def main():
    """Testing file to show neural network can learn linearly separable
    data."""
    data = np.genfromtxt("output.txt", delimiter=',')
    num_features = len(data[0]) - 1  # Subtract one because of target values

    nn = ClassifierNeuralNet(num_features)

    # NOTE: We have to wrap every target value into a vector, for the
    # purpose of being able to classify vectors later.
    nn.train(data[:,:-1], tuple((val,) for val in data[:,-1]))
    print "Done with training"

if __name__ == "__main__":
    main()
