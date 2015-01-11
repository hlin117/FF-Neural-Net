# The Lemondetection Dataset

This is a special dataset adapted from one of Kaggle's previous data science
contests, called <a href=https://www.kaggle.com/c/DontGetKicked>
"Don't Get Kicked!"</a>. The premise is to detect whether a car is a "lemon" 
or not using their training and testing datasets, downloadable from <a 
href=https://www.kaggle.com/c/DontGetKicked/data>this link</a>.

## Data Preprocessing

Data preprocessing has already been performed by 
<a href=https://github.com/peterlvilim>Peter Vilim</a>. Please speak to 
me for the data preprocessing scripts.

The results of these preprocessing scripts are "lemon\_training.csv" and
"lemon\_testing.csv", located in the data folder.

## Set Up
If you haven't already, run the following script in the root of
this project:
```bash
./link.sh
```

From there, fire away with
```
./execute.sh
```

The output of this executable is `output.csv`. You could have Kaggle
judge this dataset <a href=https://www.kaggle.com/c/DontGetKicked/submissions/attach>here</a>.
