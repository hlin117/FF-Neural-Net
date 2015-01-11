#!/usr/bin/python
from argparse import ArgumentParser
import numpy as np

"""A small python file that defines functions to automatically generate
test data for binary classification."""

def generate(l, m, n, numInstances):
    """Generates data using a uniform distribution. 
    
    Returns a num_instances-by-n matrix of samples, and a vector of
    target tags (1 for positive, -1 for negative).  In the sample matrix, 
    each row represents a samples, and each column represents a feature 
    of the sample. The resulting dataset will have 50% positive and 50%
    negative examples.

    The arguments are as follows:
    l:  Number of relevant features in the dataset. A sample needs at
        least l 1's within the first m features to be tagged positive.
    m:  Indicates the index after which the last relevant feature could exist.
    n:  The number of features in the dataset.

    In other words, the l relevant features must lie between indices 
    0 < m.  The generator makes the assumption that the predictive 
    model does not know that the features must lie within a certain range.

    The resulting data is linearly separable; the output data is not non-linear.
    """

    if not (0 < l <= m and l <= m <= n):
        raise Exception("Invalid parameters to create dataset.")

    # The 2 below is just an arbitrary baseline.
    if l < 2:
        raise Exception("l value is too small")

    numPos = numInstances // 2
    numNeg = numInstances - numPos

    pos_features, pos_targets = gen_examples(l, m, n, numPos, True)
    neg_features, neg_targets = gen_examples(l, m, n, numNeg, False)

    y = np.vstack((pos_targets, neg_targets))
    x = np.vstack((pos_features, neg_features))

    # Permute the dataset
    indexer = np.arange(numInstances)
    np.random.shuffle(indexer)
    y = y[indexer, :]
    x = x[indexer, :]

    return x, y

def gen_examples(l, m, n, numExamples, tag):
    """Returns a matrix of tagged data"""
    sign = 1 if tag else - 1
    targets = sign * np.ones((numExamples, 1))

    zerosBlock = np.zeros((numExamples, m))
    randBlock = (np.random.rand(numExamples, n - m) < 0.5) * 1
    features = np.hstack((zerosBlock, randBlock))
    numNonZeros = l if tag else l - 2

    for i in xrange(numExamples):
        features[i, :numNonZeros] = 1
        np.random.shuffle(features[i, :m])

    return features, targets

def parse_args():
    """Parsers the arguments from the command line."""
    parser = ArgumentParser(description="Generate some uniformly distributed \
            test data. Data is saved as a numInstances-by-(numFeatures + 1) \
            matrix in the specified text file.")
    parser.add_argument("l", type=int, help="Number of features needed \
            to be tagged positive")
    parser.add_argument("m", type=int, help="Max index that relevant \
            features can appear.")
    parser.add_argument("n", type=int, help="Number of features per sample")
    parser.add_argument("numInstances", type=int, help="Number of samples \
            sought from the dataset")
    parser.add_argument("--split", help="Choose to save outputs and \
            targets in different files", action="store_true")

    parser.add_argument("file", type=str, help="Output file name")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    samples, targets = generate(args.l, args.m, args.n, args.numInstances)
    if args.split:
        np.savetxt(args.file + "_features.csv", samples, delimiter=",", fmt="%.0f")
        np.savetxt(args.file + "_targets.csv", targets, delimiter=",", fmt="%.0f")
    else:
        np.savetxt(args.file, np.hstack((samples, targets)), delimiter=",", fmt="%.0f")

if __name__ == "__main__":
    main()
