# Physical-Activity-Prediction

This is the preproccesing script for the biometric data generated from the Bio-Monitor vest.
ftp://ftp.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/Bio-Monitor/

The script stiched together the biometrics into a single csv file.
Labeling the combined biometrics from multiple days with the corresponding activity (found in the Meta_data_activities.csv in the linked dataset) matched between the activities time interval.

ie: biometrics with timestamps 7:00 to 7:06 would be labled with that actvity. 

A model to predict if a subject is walking, run/jog, biking, working on computer, TV, sleep, or eating was created using the Xgboost library.

The model achived an accuracy_score of 91.98% in predicting the 7 categories.
This is based on 30% test, 70% train data.

## Main challenges:
  - Biometrics across multiple csv files.
  - Stitching together the biometrics and labeling with correct activity based on time interval
  - Activity meta data annotated using natural language with many sparse categories

