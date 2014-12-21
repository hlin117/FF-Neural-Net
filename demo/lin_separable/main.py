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
    nn = ClassifierNeuralNet(num_features, max_epochs=2,
            learn_rate=.85, scale=0.1, verbose=True)

    for i in xrange(10):
        print "Starting to train..."
        start = time()
        # NOTE: We have to wrap every target value into a vector, for the
        # purpose of being able to classify vectors later.
        targets = tuple((val,) for val in data[:,-1])
        features = data[:,:-1]

        nn.train(features, targets)
        print "Done with training. Took {0} seconds to train." \
                .format(round(time() - start, 2))

        print "Beginning with scoring..."
        start = time()
        scored_data = np.genfromtxt("output.txt_features", delimiter=",")
        correct = np.genfromtxt("output.txt_targets", delimiter=",")
        prediction = nn.score_data(scored_data)
        print "Done with scoring. Took {0} seconds to score the dataset" \
                .format(round(time() - start, 2))

        print "Total number incorrect: {0}".format(sum(1 for i in \
                xrange(len(correct)) if correct[i] != prediction[i]))


if __name__ == "__main__":
    main()
