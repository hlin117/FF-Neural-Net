#!/bin/bash

if [ ! -f demo/.gitignore ]; then
	touch ./demo/.gitignore
	echo "Created ./demo/.gitignore"
fi

# Creates hard links for the neural net model from /model to /demo
# TODO: Just write a recursive function for this instead of manually
# linking each one...
lin_sep=("classneuralnet.py" "neuralnet.py")

for i in "${lin_sep[@]}"; do
	if [ -f demo/lin_separable/$i ]; then
		continue
	fi
	ln model/$i demo/lin_separable/$i
	echo $i >> demo/.gitignore
done

echo "Done linking. The neural network is ready to be demo-ed."
echo "Appropriate files have been added to /demo/.gitignore."
