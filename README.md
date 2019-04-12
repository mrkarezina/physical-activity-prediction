# Physical-Activity-Prediction

This repository contains the script used to process the the biometric data generated from the Bio-Monitor vest. It also contains the training code for a model to predict the physical activity from the biometrics.

Data: ftp://ftp.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/Bio-Monitor/

### preprocessor.py
This script stitched together the biometrics into a single csv file. It labels the combined biometrics from multiple days with the corresponding activity (found in the Meta_data_activities.csv in the linked dataset). The Meta_data_activities dataset provides the end time of the activity. All of the biometrics from the previous activity up until this endtime are labeled with this activity.

ie: biometrics with timestamps 7:00 to 7:06 would be labeled an activity such as walking.

### combine_processed.py
Takes the pandas dataframe of labeled biometrics and and sets an activity_session id to each row. An activity session is a successive series of activities. An example would be walk -> jog -> run -> walk. This script is meant to be used to generate a dataset for training an LSTM. This LSTM can be used to predict the transitions between activities using the sequential data.

### gradient_boosting_model.py
I used the Xgboost library to make a model to predict if a subject is walking, run/jog, biking, working on computer, TV, sleep, or eating.

The model achieved an accuracy_score of 91.98% in predicting the 7 categories.
The model was trained using 30% test and 70% train data.

## Main challenges
 - Biometrics across multiple csv files.
 - Stitching together the biometrics and labeling with correct activity based on time interval
 - Activity meta data annotated using natural language with many sparse categories
