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

### Fitting Graph and Conclusions Made:
Based on the following fitting graph: ![img](image.png)

We have determined that our model is underfitted due to the similarities between the testing error and the training error. This is also supported by our decision boundaries from our SVM are warped and do not have distinct boundaries between different features. These decision boundaries were made between `n_instruments` and the other features of our dataset. 

Some things we have considered for the next training step: 
- Change the degree we used in polynomial expansion to see if a higher expansion would be beneficial to our model
- Giving the classes a weight or removing the issue of class imbalance
- randomizing our initial dataset more; `Pop_Rock` seems to be the primary genre in the dataset used in the first training step. 

### Possible next model:
From the results we have collected, we are considering implementing a decision tree because this model ignores class imbalance which fits our problem. 

## Milestone 4
Going into Milestone 4 our goals were to reduce the number of features and polish up the preprocessing as well find a way to deal with the class imabalance present in our data. We also trained several other alternative models from the SVM model we developed in Milestone 3.

### Preprocessing
In this milestone, we incorporated more non-numerical features. This involved aggregating the statistics. Features like note_counts, avg_note_durations, and avg_velocities were represented as the mean, standard deviation, and sum. These preprocessing techniques will help answer the questions like which songs have high note counts or which songs have the longest note durations. Each of these features may help improve the model's prediction on the type of genre the song is. Another preprocessing technique we have added is one-hot encoding. This technique was applied to program numbers which represents the different types of instruments played within the song file. This feature may be important to our model because certain instruments are strong indications on the song's genre. Instruments like drums, electric guitars, and bass are strong indication of a song possibly being pop rock or string instruments hinting towards a classical song. 
We also applied the min-max scalar this time around rather than the standard scalar because our features like note_counts are uniformly distributed. This means that by using Min-Max scaling transforming values to a specific range does not alter the original distribution of our features. Also Min-Max scaling is helpful for the current model we are using, that being KNN since these algorithms depend on the distances between data points. Min-max also doesn't assume a gaussian distribution which is versatile for our dataset of unique features.
Finally our last applied preprocessing was overscaling the less represented data. It turns out after looking at our genre distribution of songs, majority of our songs in our dataset were pop rock. This meant that the model would just assume majority pop rock rather than seeing the certain patterns that lead to other genres. By increasing the representation of other less represented data like classical or electric and balancing our data, we give our model a chance to generalize better.

### Models
For the MS4 we tried 2 seperate models. We tried a KNN model using oversampling due to our data having a heavy class imbalance but that ended up not being successful due to issues with the class balance being too significant and an overabundance of features. Furthermore, even though we upsampled our underrepresented data, it seems that our dataset that consists of genres other than pop rock is too small. Models like KNN and even decision trees might fail to generalize for genres with high variance in style.

The 2nd model we tried was a decision tree that used oversampling and a train-test split of 80-20. This gave us a testing accuracy of 59%. We believe that the model predicted so poorly mainly because of the features provided by the dataset not being sufficiently discriminative. The features we have might also be too much for our model to where some features are causing our model to generalize incorrectly due to overlap between genres. In regards to features, we attempted to add, change, and drop features that seemed to make no sense in classifying the midi file. Examples of these are dropping the "drum_pattern" feature, adding time signatures, and changing keys to be the first key rather than a list of all keys used in the song. What we found was that they made little to no change to the resulting models and sometimes resulting in a worse output for our models. 

### Training Error vs. Testing Error
- Training MSE: 0.07
- Testing MSE: 9.66
- Training MAE: 0.02
- Testing MAE: 1.80

![img](image.png)
Based on the following training and testing error this places our model at the end of the graph with there being a very large difference in our training error and testing error. It seems that our training error is extremely low which suggests that the model fits the training data perfectly which indicates a deep decision tree or overly complex KNN. On the other hand our testing error is significantly higher than our training error which shows a high indication of poor generalization. With low training error and high testing error it is a strong sign that our graph is being heavily overfitted. It seems that our model could be capturing a noise in the training data that doesn't translate over to the unseen data. Our model is overly flexible, fitting the training data but failing to generalize to the testing data.

### Next Models
Feature reduction needs to be heavily implemented as it is likely that the amount of features we have is causing a significant amount of noise in the models that is throwing them off. There needs to be some way that accounts for feature overlapping between song genres. Possible other models are neural networks due to their ability to handle complex datasets. Neural Networks could help in capturing the complex patterns and nonlinear relationships in data. Music genres with nonlinear relationships between features like tempo, instruments, and melodic intervals can be modeled more intricately by neural networks compared to KNN and decision trees. Given a subtle interaction between tempo and the types of instruments to indicate a classical genre may be caught by the neural network model but not other traditional algorithms.

### Conclusion
Between model 1 and model 2 not many improvements were able to be made. Model 1 had an accuracy of 63% while Model 2 has an accuracy over 59% they are also both heavily guessing the dominant class of Pop_Rock rather than any other class. The overall accuracy went down and even accounting for the class imabalance the models do not seem to be improving. Both models have overfitting issues which still remain unresolved. In conclusion Model 2 has failed to make improvements and has actually gotten a little worse compared to the previous model. It seems that the issue could possibly be coming from the dataset not be discriminative enough for both our models to accuractly distinguish between song genres. A feature like tempo can vary by genre. Even though electronic songs have faster beats compared to ballads, other genres may overlap with these genres which makes it difficult for the model to use a tempo feature to differentiate between the songs. This is similar with other of our genres like average note durations and variability being too general to differentiate between genres. Another example is key modulations. Although some genres are known for frequent modulations like classical or rare modulation like pop rock. Not all genres follow these rules so there again tends to be a lot of overlap between genres. Finally our biggest issue is that our dataset has a genre called pop rock which dominates our dataset. Not only is the genre being overrepresented but pop rock seems to be such a general genres that has multiple overlapping features. Although some pop rock songs follow similar tempo and variability, there are also many other songs that fall in the pop rock genre with really low tempo or really high tempo. The overrepresented genre that overlaps with many of our other genres makes it hard for our model to make the correct predictions.

Finally the reason why model 1 might be doing better than model 2 could be due to the svm being a better fit for our model since the number of features we have makes it overly complex. Reducing the number of features a little more may allow the decision tree to improve. Another reason could just be the random sample of data trained and tested by the KNN and descision tree is more biased than the sample of data used for the SVM. It should be kept in mind that we are using a randomized sample size of the actual whole Lakh Midi dataset because the dataset is too large and the front end of the dataset is predominantly pop rock.

### Correct Predictions, FP and FN
From our test dataset, an example of a correct prediction (true positive) our model made was that a MIDI file's genre is pop-rock, Jazz, and Electronic. An example of a false negative is that an electronic song was incorrectly classified as a pop-rock song, which makes it a false negative for our electronic genre because the MIDI file was actually electronic but was classified as pop rock. An example of a false positive is that a song that was supposed to be a pop-rock was classified as an electronic song, which makes it a false positive for the electronic song label since the actual genre isn't the same. These are all represented in the last block of code represented as lists of predicted vs actual data.
