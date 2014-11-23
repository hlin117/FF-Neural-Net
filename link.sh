#!/bin/bash

models=("classneuralnet.py" "neuralnet.py")
folders=("lin_separable" "xor_dataset")

# For the case that we just want to remove all of the generated
# links and the .gitignore
if [ "$1" = "reset" ]; then

	if [ -f demo/.gitignore ]; then
		rm demo/.gitignore
	fi

	for folder in "${folders[@]}"; do
		for model in "${models[@]}"; do
			if [ -f demo/$folder/$model ]; then
				rm demo/$folder/$model
			else
				echo "Couldn't find file demo/$folder/$model"
			fi
		done

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

for folder in "${folders[@]}"; do
	for model in "${models[@]}"; do
		if [ -f demo/$folder/$model ]; then
			continue
		fi
		ln model/$model demo/$folder/$model
		echo $folder/$model >> demo/.gitignore
	done
done

echo "Done linking. The neural network is ready to be demo-ed."
echo "Appropriate files have been added to /demo/.gitignore."
