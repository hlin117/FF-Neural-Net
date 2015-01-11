Linearly Separable Data
=======================

This series of tests has the neural network learn from linearly
separable data.


Set up
======
If you haven't already, run the following script in the root of
this project:
```bash
./link.sh
```
The script to create testing data is located in the project's
data folder. To generate a training file here, use the following:
```bash
./data/gen_linear.py 5 10 15 20000 training
```

This will create a file called `data` in this directory,
with 5 relevant features between indices 0 to 10 (exclusive),
each sample containing 20 features. This dataset will contain 
2000 training samples.

To generate a complementing testing set, use the following:
```bash
./data/gen_linear.py 5 10 15 200 testing --split
```
IDEAL PARAMETERS
================
TODO: Still trying to search.
