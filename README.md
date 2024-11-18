# CSE 151A Group Project <a target="_blank" href="https://colab.research.google.com/drive/19ArXW2768P2VxWBXff0s5CjbLbCtjqJk?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

## Milestone 2

### Initial Data Exploration:
The dataset contains 16 different subdirectories, each containing different audio samples. There is a total of 176,581 entries in the dataset, and upon looking at the dataset we noticed that many of the files were corrupted. Using the `pretty-midi` utility functions, we can extract the different attributes of the dataset. For the purposes of this project, we will be focusing on the following components: 
1.  `n_instruments`: the number of unique instruments in the file
2.  `tempo_estimate`: the estimated tempo of the file
3.  `program_numbers`: the unique integer value corresponding to the type of instrument used in the file. The list of program numbers can be found [here](https://midiprog.com/program-numbers/)
4.  `key numbers`: list of key value integers which represent the key signatures in the file
5.  `time_signature_changes`: list containing time signature changes in file (e.g.: 4/4 represents 4 beats in each measure (denoted by numerator) and the note value that represents 1 beat (denoted by denominator). In this example the 4 quarter-note beats is commonly used in many genres like classical, pop, and rock.)
6.  `end_time`: duration of the music piece
7.  `tempo`: a list of tempos (beats per minute) found in the file

### Preprocessing:
As mentioned in the data exploration section we have noticed that a lot of the entries contain corrupted audio files. We have decided to drop these entries as the dataset already contains a lot of entries. The files should already follow the [Standard MIDI Format](https://majicdesigns.github.io/MD_MIDIFile/page_smf_definition.html#:~:text=Standard%20MIDI%20File%20Format,page%20authoring%20and%20greeting%20cards.), so we do not need to do any additional standardization. We will not implement any encoding methods since all of the values we will be examining are continuous values.   


## Milestone 3

### Preprocessing:
Before we began preprocessing the data, we first had to get the ground truth `genres`. In order to get our ground truth, we had to cross-reference the ID of the associated MIDI file with the following [link](https://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-MAGD-genreAssignment.cls). We also used the helper function `compute_statistics(midi_file)` to get all of the features from a midi file to create our data frame, and any pieces of data that had any null values would be dropped from our dataset. From there we imputed both categorical and discrete features and scaled it to your then perform polynomial expansion. 

### Training:
For the training of our dataset, we first decided to use two support vector machines, a linear SVM and RBF SVM. Measuring the error of our model using MSE (Mean Squared Error), we got the following:  
- Training Accuracy Linear MSE: 13.48
- Testing Accuracy Linear MSE: 21.03
- Training Accuracy Linear ACC: 0.69
- Testing Accuracy Linear ACC: 0.52
- Training Accuracy RBF MSE: 13.48
- Testing Accuracy RBF MSE: 14.19
- Training Accuracy RBF ACC: 0.69
- Testing Accuracy RBF ACC: 0.63

### Fitting Graph and Conclusion:
For our fitting graph, we created a decision boundary, and we primarily looked at the relationship between `n_instruments` and the other categories. The boundaries within the images that were produced seemed to be warped and not organized. This along with the fact that our errors measured were very close to each other suggests that our model is on the underfitting side. 

Some things we have considered for the next training step: 
- Change the degree we used in polynomial expansion to see if a higher expansion would be beneficial to our model
- Giving the classes a weight or removing the issue of class imbalance
- randomizing our initial dataset more; `Pop_Rock` seems to be the primary genre in the dataset used in the first training step. 

### Possible next model:
From the results we have collected, we are considering implementing a decision tree because this model ignores class imbalance which fits our problem. 

