#!/usr/bin/python
import csv

with open('results.txt', 'r') as nnFile:
    with open('raw_testing.csv', 'r') as testingFile:
        predictionsNN = csv.reader(nnFile)
        testingData = csv.reader(testingFile)
        for lineNN, lineTest in zip(predictionsNN, testingData):
            if lineTest[0] == "RefId":
                print lineTest[0]+","+lineNN[0]
            else:
                print lineTest[0]+","+lineNN[0]
