#!/usr/bin/python
import numpy as np
from classneuralnet import ClassifierNeuralNet


def main():
    """Testing file to show neural network can learn linearly separable
    data."""
    data = np.genfromtxt("output.txt", delimiter=',')
    num_features = len(data[0]) - 1  # Subtract one because of target values

    nn = ClassifierNeuralNet(num_features)
    nn.train(data[:,:-1], data[:,-1])




if __name__ == "__main__":
    main()
