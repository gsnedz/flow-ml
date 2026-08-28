# So You Want to Use Machine Learning to Calculate Elliptical Flow:
### Flow ML Documentation
### Garrett Snedden, Summer 2026

Hi!  If you’re reading this, it means you probably want to train neural nets to do flow calculations!  This code contains a lot of that- DNNs, datasets to train and test them, code to test them, all that good stuff.

## Main Findings of the Project
### Nets
We set out to apply machine learning to flow calculations, and found that DNNs could learn and efficiently calculate Q-vectors, which are then used to calculate correlators.  Calculations for Q-vectors, correlators, and cumulants can be found in this paper: https://arxiv.org/pdf/1010.0233.  We created DNNs which calculate x- and y-components of
Q2, Q4, Q6, Q8 (for use in calculating c2{2}, c2{4}, c2{6}, and c2{8}).
Qa, Qb, Qc, Qd (for use in calculating c2{2} by splitting each event into four subevents).
Q2, Q4, Qa, Qb, Qc, Qd (combining both of the above).
p2, q2, q4, Q2, Q4 (for calculating d2{2} and d2{4} for differential flow) (not yet working)

### Time-saving
From calculating the amount of time it takes to run truth calculations when creating the datasets to run the data through the NNs, we found that running data through the nets was far faster than truth calculations.  This does not include all the saving and loading of arrays, which takes the bulk of time in both truth and net calculations.

### Data Rotation
In creating the datasets, there is an option to rotate each event by some angle (or multiple angles) ⍺, such that each 𝜙 in the event undergoes the transformation 𝜙 → 𝜙 + ⍺.  Adding an additional copy of the dataset rotated greatly increases the training efficiency, even with a few rotations, as seen below with a starting dataset size of 50,000 events.  In cases where data is limited, rotating the data can help to simulate a larger dataset.


## Using the Code
### Creating a Dataset
To create a dataset, use the create_dataset() function in deepsets.py.  The “path” parameter specifies the folder under which the dataset is saved.  Inputting a “chunk_size” argument will save the dataset under multiple datasets, each of the specified size “_i” appended to the name for i-many chunks.
Note: if creating datasets to test a NN for differential flow, name each dataset “{name}_{i}” for i many pt bins.  Yeah, the notation for chunks and differential flow is the same.  Don’t use them at the same time.

### Training a NN
To train a neural net, use the train_net() function in dnn.py.  The “folders” parameter can be a string or list of strings specifying the dataset(s) to be trained on.  If the dataset was saved in chunks and the “chunks” parameter here is set to True, the net will automatically train on all saved chunks.  The net saves to the models folder under the same name(s) of the dataset(s) used to train it.  I trained nets of the following varieties.
Q2, Q4, Q6, Q8 (for use in calculating c2{2}, c2{4}, c2{6}, and c2{8}).  This net has 5 inputs (phi, eta, pt, cos(phi), sin(phi)) and 8 outputs (x- and y-components for each Q-vector).

Qa, Qb, Qc, Qd (for use in calculating c2{2} by splitting each event into four subevents).  This net has 5 inputs and 8 outputs (same as above).
Q2, Q4, Qa, Qb, Qc, Qd (combining both of the above).  This net has 5 inputs (same as above) and 12 outputs (x- and y-components for each Q-vector).

p2, q2, q4, Q2, Q4 (for calculating d2{2} and d2{4} for differential flow).  This has 7 inputs (the 5 inputs above followed by one-hot embedding labelling particles as POI or reference particles).

### Testing the NN
There are several files labelled “testing”: testing.py, test_all.py, test_all.py, test_differential.py, which create cumulant graphs for the above NNs, respectively.
NOTE: in subevents.py, there’s an statement “from calculations import *”.  Because of slurm files in different folders, I’ve had to change that back and forth between

	from calculations import *
for deepsets.py and

	from deepsets.calculations import *
for testing files..  I never figured out how to fix that error with slurm.  If you figure out how to fix this, power to you.

### Most Important Functions
deepsets.py - main().  Creates datasets.
dnn.py - main().  Trains and saves NN.
testing.py - load_data().  Sends data through the NN and saves the results or loads results from a previously processed dataset.
	plot_corr().  Graphing function.
test_subevents.py - test_subevents().  Graphing function.
test_all.py - test_all().  Graphing function.
test_differential.py - plot_differential_corrs().  Graphing function.

### File Folders
The file paths coded into the system assume the following structure.

> main folder
Contains all testing python files, dnn5.py, as well as all slurm files.
> arrays
When datasets are run through the NNs, the results are stored here for easy access
> deepsets
Contains all files relevant for creating datasets.  Namely, add_flow.py, calculations.py, deepsets.py, differential.py, and subevents.py. 
> Data
Contains all root files
> deepset_data
When datasets are created, they are stored here.
> flow_arrays
When flow is added to a dataset during creation, the result is stored here so that flow does not need to be recreated if the dataset is recreated.
> dnn
Contains dnn.py, the file containing classes for the NNs.
> images
Stores graphs plotting the results of various NNs.
	> combined
	Stores the graphs from NNs that output Q-vectors for both no subevents and 4 subevents.
	> differential
	Stores graphs from NNs that output Q-vectors for differential flow.
	> subevents
	Stores graphs from NNs that output Q-vectors for 4 subevent calculations.
