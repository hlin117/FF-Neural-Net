#!/bin/bash

if [ ! -f link.sh ]; then
    echo "Error: Cannot execute if shell is not in the same directory as script"
    exit 0
fi

models=("neuralnet.py")
folders=("lin_separable" "xor_dataset" "lemondetection" "load_NN" "small_testing")

# For the case that we just want to remove all of the generated
# links and the .gitignore
if [ "$1" = "reset" ]; then

    
	for folder in "${folders[@]}"; do
		for model in "${models[@]}"; do
            file1=demo/$folder/$model
            file2=model/$model
			if [ -f $file1 ]; then

                hash1=`md5 $file1`
                hash2=`md5 $file2`

                if [ "$hash1" = "$hash2" ]
                then
                    echo "Files have NOT the same content. Will not remove."
                    exit 0
                fi

				rm $file1
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
done

echo "Done linking. The neural network is ready to be demo-ed."
echo "Appropriate files have been added to /demo/.gitignore."
