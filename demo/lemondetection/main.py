#!/usr/bin/python
import csv
import sys
from neuralnet import NeuralNet
from time import time
import numpy as np
from random import shuffle

def main():

    print "Loading in the data..."
    text = open("training.csv").read().split("\n")
    data = list(map(int, sample.strip().split(",")) for sample in text
                if sample.strip() != "")
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
#    testing = np.genfromtxt("testing.csv", delimiter=",")
    
    scored_text = open("testing.csv").read().split("\n")
    testing = list(map(int, sample.strip().split(',')) for sample in scored_text
                   if sample.strip() != "")
    predictions = nn.score_data(testing)
    print "Done with scoring. Took {0} seconds to score the dataset" \
            .format(round(time() - start, 2))

    with open("results.txt", "w") as f:
        for pred in predictions:
            f.write(str(pred) + "\n")


if __name__ == "__main__":
    main()
