#!/usr/bin/python
import numpy as np
from classneuralnet import ClassifierNeuralNet

def main():
    """Testing file to show neural network can learn linearly separable
    data."""
    data = np.genfromtxt("output.txt", delimiter=',')
    print data




if __name__ == "__main__":
    main()
