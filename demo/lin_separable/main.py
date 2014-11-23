#!/usr/bin/python
import sys
from time import time
import numpy as np
from classneuralnet import ClassifierNeuralNet


def main():
    """Testing file to show neural network can learn linearly separable
    data."""
    data = np.genfromtxt("output.txt", delimiter=',')
    num_features = data.shape[1] - 1  # Subtract one because of target values
    nn = ClassifierNeuralNet(num_features, learn_rate=0.5, default_bias=0)


    print "Starting to train..."
    start = time()
    # NOTE: We have to wrap every target value into a vector, for the
    # purpose of being able to classify vectors later.
    nn.train(data[:,:-1], tuple((val,) for val in data[:,-1]))
    print "Done with training. Took {0} seconds to train.".format(time() - start)

    print "Beginning with scoring..."
    start = time()
    data = np.genfromtxt("output.txt_features", delimiter=",")
    correct = np.genfromtxt("output.txt_targets", delimiter=",")
    prediction = nn.score_data(data)
    print "Done with scoring. Took {0} seconds to score the dataset".format(time() - start)
    print "Total number incorrect: {0}".format(sum(1 for i in \
            xrange(len(correct)) if correct[i] != prediction[i]))

    with open("results.txt", "w") as f:
        for guess in prediction:
            f.write(str(guess) + "\n")


if __name__ == "__main__":
    main()
