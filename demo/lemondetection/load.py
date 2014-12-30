#!/usr/bin/python
from time import time
from neuralnet import NeuralNet

"""Script to test out loading a neural network from a json file"""
nn = NeuralNet.load("weights2.json")

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
        f.write(str(pred[0, 0]) + "\n")

