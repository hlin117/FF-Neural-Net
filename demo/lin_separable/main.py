#!/usr/bin/python
import sys
from time import time
import numpy as np
from random import shuffle
from neuralnet import NeuralNet

def main():
    """Testing file to show neural network can learn linearly separable
    data."""
    data = np.genfromtxt("training.csv", delimiter=',').tolist()

    shuffle(data)

    # NOTE: We have to wrap every target value into a tuple, for the
    # purpose of being able to classify n-tuples later
    targets = list((sample[-1] if sample[-1] == 1 else 0,) for sample in data)
    features = list(sample[:-1] for sample in data)
    print "Starting to train..."
    start = time()

    num_features = len(features[0])  # Subtract one because of target values
    nn = NeuralNet(num_features, max_epochs=2, default_bias="random",
                   learn_rate=.85, scale=0.1, verbose=True)
    nn.train(features, targets)
    print "Done with training. Took {0} seconds to train." \
            .format(round(time() - start, 2))

    print "Beginning with scoring..."
    start = time()
    scored_data = np.genfromtxt("data_features.csv", delimiter=",")
    correct = np.genfromtxt("data_targets.csv", delimiter=",")
    prediction = nn.score_data(scored_data)
    print "Done with scoring. Took {0} seconds to score the dataset" \
            .format(round(time() - start, 2))
    num_incorrect = sum(1 for i in xrange(len(correct)) \
                        if correct[i] != prediction[i])
    print "Total number incorrect: {0}".format(num_incorrect)


if __name__ == "__main__":
    main()
