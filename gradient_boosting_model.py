import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# If not labels match
default_label = 0

train_model = False

# Which subject to exlude from train, and test on
exclude_subject = "D"

save_model = "xgmodel.pickle.dat"
save_prediction = "prediction_d.csv"

combined_biometric_data = "combined_biometric.csv"

activities = [
    {
        "substring": ["walk"],
        "numerical": 1
    },
    {
        "substring": ["jog", "run"],
        "numerical": 2
    },
    {
        "substring": ["bike", "cycle"],
        "numerical": 3
    },
    {
        "substring": ["computer", "work"],
        "numerical": 4
    },
    {
        "substring": ["movie", "tele", "watch"],
        "numerical": 5
    },
    {
        "substring": ["sleep"],
        "numerical": 6
    },
    {
        "substring": ["eat", "breakfast", "lunch", "dinner"],
        "numerical": 7
    }

]


def check_activities(activity):
    """
    Checks if substring in activity
    :param activity:
    :return:
    """

    for substrings in activities:
        for sub in substrings["substring"]:
            if sub in activity.lower():
                return True, substrings["numerical"]

    return False, default_label


def label_data(data):
    """
    Used to label the activites, by substring matching
    :param data:
    :return:
    """

    for i in data.index:
        activity = data.at[i, 'Activity']

        # Using the general activity name
        is_valid, activity = check_activities(activity)
        data.at[i, 'Activity'] = activity

    return data


def drop_data(data_frame):
    """
    Drops not needed column
    :param data_frame:
    :return:
    """

    data = data_frame.drop("time", axis=1)
    data = data.drop("subject", axis=1)
    data = data.drop("day", axis=1)
    data = data.drop("Unnamed: 0", axis=1)

    return data


def prepare_arrays(data_frame, test_size=0.3):
    """
    Creates nd arrays of data with numerical labels
    :param data_frame:
    :return:
    """

    x = data_frame[["heart_rate", "tidal_volume_adjusted", "cadence", "step", "activity", "temperature_celcius",
                    "systolic_pressure_adjusted", "minute_ventilation_adjusted"]]

    y = pd.to_numeric(data_frame['Activity'], errors='coerce').astype(float)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=True)

    print("Train shape: {0}".format(X_train.shape[0]))

    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    return X_train, X_test, Y_train, Y_test


def predict_activity(my_model, X_test, Y_test):
    """
    Use model to make prediction and save
    :param my_model:
    :param X_test:
    :param Y_test:
    :return:
    """

    print("Testing")
    predictions = my_model.predict(X_test)

    print("Accuracy Score: {0}".format(accuracy_score(predictions, Y_test)))
    print("Variance: {0}".format(explained_variance_score(predictions, Y_test)))

    pred = ["Prediction", "Truth", "Heart Rate"]
    cleaned = pd.DataFrame(columns=pred)
    cleaned["Prediction"] = list(predictions)
    cleaned["Truth"] = list(Y_test)
    cleaned["Heart Rate"] = [X[0] for X in X_test]

    cleaned.to_csv(save_prediction)


data = pd.read_csv(combined_biometric_data)
data = data.query("Activity.notnull()")

# Data with all subjects
all_subject_data = label_data(data)

# Dont include default category
all_subject_data = all_subject_data.query("Activity != 0")

# Exlude a subject
excluded_subject = all_subject_data.query("subject != \"D\"")
# exlude_subject = all_subject_data.copy()

X_train, X_test, Y_train, Y_test = prepare_arrays(excluded_subject)

if train_model:
    eval_set = [(X_test, Y_test)]
    my_model = XGBClassifier(n_estimators=500, learning_rate=0.05, verbose=True)

    print("Training")
    # X_train = list(X_train)
    # Y_train = list(Y_train)
    # two = zip(X_train, Y_train)
    # for t in two:
    #     print(t)

    my_model.fit(X_train, Y_train, verbose=True, eval_set=eval_set)
    results = my_model.evals_result()
    print(results)

    pickle.dump(my_model, open(save_model, "wb+"))
else:
    my_model = pickle.load(open(save_model, "rb"))

predict_activity(my_model, X_test, Y_test)

"""
Testing for particular subject
"""
print("Testing select: ")
qeury = "subject == \"{0}\"".format(exclude_subject)
data = all_subject_data.query(qeury)
print("Subject test {0}".format(data.shape[0]))

data = drop_data(data)

X_train, X_test, Y_train, Y_test = prepare_arrays(data, test_size=0.3)
predictions = my_model.predict(X_test)

print("Accuracy Score: {0}".format(accuracy_score(predictions, Y_test)))
print("Variance: {0}".format(explained_variance_score(predictions, Y_test)))

pred = ["Prediction", "Truth", "Heart Rate"]
cleaned = pd.DataFrame(columns=pred)
cleaned["Prediction"] = list(predictions)
cleaned["Truth"] = list(Y_test)
cleaned["Heart Rate"] = [X[0] for X in X_test]

cleaned.to_csv('single_' + save_prediction)
