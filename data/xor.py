#!/usr/bin/python
import itertools
import random
import json

# Generates the truth table for xor
table = list(itertools.product([1, 0], repeat=12))
targets = list(reduce(lambda a, b: a ^ b, row) for row in table)
targets = list(1 if val == 1 else -1 for val in targets)
results = list()
for i in xrange(len(table)):
	innerlist = list(table[i])
 	innerlist.append(targets[i])
 	results.append(innerlist)
 
random.shuffle(results)
with open("xor_data.txt", "w") as f:
    json.dump(results, f, indent=4)
    
