#!/usr/bin/python
import sys
from time import time
import numpy as np
from random import shuffle

def main():
    """Testing file to show neural network can learn linearly separable
    data."""
    data = np.genfromtxt("output.txt", delimiter=',').tolist()


    for i in xrange(10):
        shuffle(data)

        # NOTE: We have to wrap every target value into a tuple, for the
        # purpose of being able to classify n-tuples later
        targets = np.array(tuple((sample[-1],) for sample in data))
        features = tuple(sample[:-1] for sample in data)
        print "Starting to train..."
        start = time()

        num_features = len(data[0]) - 1  # Subtract one because of target values
        nn = NeuralNet(num_features, max_epochs=2,
                learn_rate=.85, scale=0.1, verbose=True)
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
