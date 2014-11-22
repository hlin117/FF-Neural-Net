#!/bin/bash

lin_sep=("classneuralnet.py" "neuralnet.py")

# For the case that we just want to remove all of the generated
# links and the .gitignore
if [ "$1" = "reset" ]; then

	if [ -f demo/.gitignore ]; then
		rm demo/.gitignore
	fi

	for i in "${lin_sep[@]}"; do
		if [ -f demo/lin_separable/$i ]; then
			rm demo/lin_separable/$i
		fi
	done
	echo "Removed ./demo/.gitignore and other links"
	exit 0
fi

# Normal case: when we want to only create the files
if [ ! -f demo/.gitignore ]; then
	touch ./demo/.gitignore
	echo "Created ./demo/.gitignore"
else
	echo "./demo/.gitignore detected. run './link reset' to remove files"
	exit 0
fi

# Creates hard links for the neural net model from /model to /demo
# TODO: Just write a recursive function for this instead of manually
# linking each one...

for i in "${lin_sep[@]}"; do
	if [ -f demo/lin_separable/$i ]; then
		continue
	fi
	ln model/$i demo/lin_separable/$i
	echo $i >> demo/.gitignore
done

echo "Done linking. The neural network is ready to be demo-ed."
echo "Appropriate files have been added to /demo/.gitignore."
