#!/bin/bash
if [ -f output_actual.csv ] ; then
    rm output_actual.csv
fi
if [ -f results.txt ] ; then
    rm results.txt
fi

./load.py
./combine.py > output_actual.csv

if [ -f results.txt ] ; then
    rm results.txt
fi
