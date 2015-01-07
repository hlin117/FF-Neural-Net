#!/bin/bash
if [ -f "results.txt" ]; then
	rm results.txt
fi

./main.py

if [ -f "results.txt" ]; then
	./combine.py > output.csv
	rm results.txt
fi
