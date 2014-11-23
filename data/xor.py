#!/usr/bin/python
import itertools
import random
import json
from argparse import ArgumentParser

def parse_args():
    """Parsers the arguments from the command line."""
    parser = ArgumentParser(description="Generates a truth table for xor. \
            Output is a json list of lists, where the last element is 1 or -1")
    parser.add_argument("n", type=int, help="Number of booleans desired \
            for the xor calculation.")
    parser.add_argument("file", type=str, help="Output file name")
    args = parser.parse_args()
    return args

def main():
    """Generates the truth table for xor"""
    args = parse_args()

    table = list(itertools.product([1, 0], repeat=args.n))
    targets = list(reduce(lambda a, b: a ^ b, row) for row in table)
    targets = list(1 if val == 1 else -1 for val in targets)

    results = list()
    for i in xrange(len(table)):
            innerlist = list(table[i])
            innerlist.append(targets[i])
            results.append(innerlist)
     
    random.shuffle(results)
    with open(args.file, "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
