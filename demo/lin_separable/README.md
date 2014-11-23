Linearly Separable Data
=======================

This series of tests has the neural network learn from linearly
separable data.


Set up
======
If you haven't already, run the following script in the root of
this project:
```bash
./link
```
The script to create testing data is located in the project's
data folder. To generate a testing file here, use the following:
```bash
chmod u+rwx ./../../data/gen_linear.py
./../../data/gen_linear.py 5 10 20 100 output.txt
```
This will create a file called `output.txt` in this directory,
with 5 relevant features between indices 0 to 10 (exclusive),
each sample containing 20 features.
