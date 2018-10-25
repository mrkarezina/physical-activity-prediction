# Physical-Activity-Prediction

This is the preproccesing script for the biometric data generated from the Bio-Monitor vest.
ftp://ftp.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/Bio-Monitor/

The script stiched together the biometrics into a single csv file.

A model to predict if a subject is walking, run/jog, biking, working on computer, TV, sleep, or eating was created using the Xgboost library.

The model achived an accuracy_score of 91.98% in predicting the 7 categories.
