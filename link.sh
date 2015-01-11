#!/bin/bash

if [ ! -f link.sh ]; then
    echo "Error: Cannot execute if shell is not in the same directory as script"
    exit 0
fi

models=("neuralnet.py")
folders=("lin_separable" "lemondetection" "load_NN")

# For the case that we just want to remove all of the generated
# links and the .gitignore
if [ "$1" = "reset" ]; then

    
	for folder in "${folders[@]}"; do
		for model in "${models[@]}"; do
            linked_file=demo/$folder/$model
            model_file=model/$model

            # Executes only if the file exists
			if [ -f $linked_file ]; then
#                diff=`diff $linked_file $model_file`
#                if [ "$diff" != "" -a "$2" != "force" ] ; then
#                    echo "File $linked_file is not the same, aborting."
#                    echo "Difference"
#                    echo $diff
#                    exit 0
#                fi

				rm $linked_file
			else
				echo "Couldn't find file demo/$folder/$model"
			fi
		done


	done

	if [ -f demo/.gitignore ]; then
		rm demo/.gitignore
	fi

	echo "Removed ./demo/.gitignore and other links"
	exit 0
fi

# Normal case: when we want to only create the files
if [ ! -f demo/.gitignore ]; then
	touch ./demo/.gitignore
	echo "Created ./demo/.gitignore"
else
	echo "./demo/.gitignore detected. run './link.sh reset' to remove files"
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

    if [ ! -L "demo/$folder/data" ]; then
        ln -s data "demo/$folder/data"
    fi

done

# Ignored files
echo *.csv >> demo/.gitignore
echo data >> demo/.gitignore

echo "Done linking. The neural network is ready to be demo-ed."
echo "Appropriate files have been added to /demo/.gitignore."
