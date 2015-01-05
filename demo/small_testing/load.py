#!/usr/bin/python
from __future__ import division
from neuralnet import NeuralNet
import numpy as np
from scipy.special import expit

"""Script to test out loading a neural network from a json file"""
nn = NeuralNet.load("item0.json")
nn.default_act = lambda x: 2 * expit(x) - 1
nn.default_deriv = np.vectorize(lambda x: 2 * x * (1 - x))

data = np.genfromtxt("small_training.csv", delimiter=',').tolist()
targets = np.array(tuple((sample[-1],) for sample in data))
features = tuple(sample[:-1] for sample in data)

nn.train(features, targets)
