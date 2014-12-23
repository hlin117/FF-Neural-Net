#!/usr/bin/python
import math
from random import shuffle
from time import time
from classneuralnet import ClassifierNeuralNet
import json


def split(data):
    """By default, the data comes as one row, with the tag of the
    sample on the same line as the features. This returns two lists
    which has the features and targets separated.
    """
    features = list()
    targets = list()

    for row in data:
        features.append(tuple(row[:-1]))
        targets.append((row[-1],))

    return features, targets


def main():
    table = json.load(open("xor_data.txt"))
    shuffle(table)
    
#    # Using 80% of the data to train, and 20% to test
#    fifth = len(table) / 5
    training = table
    testing = table

    # Creates and trains the neural network
    train_feat, train_targets = split(training)
    num_features = len(train_feat[0])

    print "Training neural network on {0} samples".format(len(train_feat))
    start = time()
    nn = ClassifierNeuralNet(num_features, verbose=True, scale=20,
                             hidden_layer=[10], max_epochs=5)
    nn.train(train_feat, train_targets)
    print "Done training. Took {0} seconds.".format(time() - start)

    # Tests the neural network
    test_feat, test_targets = split(testing) 

    print "Testing the neural network on {0} samples".format(len(test_feat))
    start = time()
    predictions = nn.score_data(test_feat)
    numincorrect = sum(1 for i in xrange(len(test_feat))
            if test_targets[i][0] != predictions[i])
    print "Done testing. Took {0} seconds.".format(time() - start)
    print "Number incorrect: {0}".format(numincorrect)


if __name__ == "__main__":
    main()
