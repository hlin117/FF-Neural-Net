Feedforward Neural Network
==========================

This is my own implementation of a feedforward neural network. It's currently under
construction, but I plan to have it support the following:
<ul>
	<li>A network with one hidden layer.</li>
	<li>Drawing of the neural network on a png image.</li>
	<li>Setting the default activation functions. (Default will be the sigmoid function.)</li>
	<li>5-fold cross validation.</li>
	<li>Threshold tuning, in the case that the neural network is performing a 
	binary classification problem.</li>
</ul>

Resources
=========
<ul>
	<li><a href="http://page.mi.fu-berlin.de/rojas/neural/">Neural Networks - A Systematic
	Introduction (Raul Rojas)</a></li>

	<li> <a href="http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw">
	Cross Validated - How to choose the number of hidden layers and nodes in a feedforward neural network?</a></li>
</ul>

Set up
======
Run the following script to create hard links between the neural network files
and the demo scripts located in /demo:
```
./link.sh
```
This will also add these links to the ./demo/.gitignore, so they will not be
duplicated under version control.
