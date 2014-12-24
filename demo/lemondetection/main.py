#!/usr/bin/python
import csv
import sys
from neuralnet import NeuralNet
from time import time
import numpy as np
from random import shuffle

def main():
    """Testing file to show neural network can learn linearly separable
    data."""
    print "Reading from the text file..."
    #data = np.genfromtxt("training.csv", delimiter=',').tolist()
    print "Shuffling..."
    shuffle(data)

    # NOTE: We have to wrap every target value into a tuple, for the
    # purpose of being able to classify n-tuples later
    targets = np.array(tuple((sample[-1],) for sample in data))
    features = tuple(sample[:-1] for sample in data)
    print "Starting to train..."
    start = time()

    num_features = len(data[0]) - 1  # Subtract one because of target values
    nn = NeuralNet(num_features, max_epochs=2, learn_rate=.85, scale=0.1, 
                   hidden_layer=[7], verbose=True)
    nn.train(features, targets)
    print "Done with training. Took {0} seconds to train." \
            .format(round(time() - start, 2))

    print "Beginning with scoring..."
    start = time()
    testing = np.genfromtxt("testing.csv", delimiter=",")
    predictions = nn.score_data(testing)
    print "Done with scoring. Took {0} seconds to score the dataset" \
            .format(round(time() - start, 2))

    with open("results.txt", "w") as f:
        for pred in predictions:
            f.write(str(pred) + "\n")


if __name__ == "__main__":
    main()
