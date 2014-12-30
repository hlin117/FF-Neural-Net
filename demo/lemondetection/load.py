#!/usr/bin/python
from __future__ import division
from time import time
from neuralnet import NeuralNet
import numpy as np
from scipy.special import expit

"""Script to test out loading a neural network from a json file"""
nn = NeuralNet.load("weights2.json")
#nn.default_act = np.vectorize(lambda x: 2 * expit(x) - 1)
nn.default_act = np.vectorize(lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x)))

print "Beginning with scoring..."
start = time()

scored_text = open("testing.csv").read().split("\n")
testing = list(map(int, sample.strip().split(',')) for sample in scored_text
               if sample.strip() != "")
predictions = nn.score_data(testing)
print "Done with scoring. Took {0} seconds to score the dataset" \
        .format(round(time() - start, 2))

with open("results.txt", "w") as f:
    f.write("IsBadBuy\n")
    for pred in predictions:
        if pred[0, 0] < 0:
            f.write(str(0) + "\n")
        else: f.write(str(pred[0, 0]) + "\n")

