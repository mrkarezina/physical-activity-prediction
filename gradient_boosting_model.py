import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# If not labels match
default_label = -1
test_size = 0.4
train_model = True

# Which subject to exclude from training. Subject will be used and test on.
test_subject = "D"

save_model = "xgb_model.pickle"
save_prediction = "predictions.csv"
combined_biometric_data = "preprocessed_biometric.csv"

selected_features = ["heart_rate", "tidal_volume_adjusted", "step", "temperature_celcius",
                     "systolic_pressure_adjusted", "minute_ventilation_adjusted"]

activities = [
    {
        "substring": ["drive"],
        "numerical": 0
    },
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
    Used to label the raw text activities, by matching substrings
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


def prepare_arrays(data_frame, test_size=0.4):
    """
    Creates nd arrays of data with numerical labels
    :param data_frame:
    :return:
    """

    x = data_frame[selected_features]

    y = pd.to_numeric(data_frame['Activity'], errors='coerce').astype(float)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=True)

    print("Train shape: {0}".format(X_train.shape[0]))

    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    return X_train, X_test, Y_train, Y_test


def evaluation(predicted, expected):
    print("Accuracy Score: {0}".format(accuracy_score(predicted, expected)))


def predict_activity(my_model, X_test, Y_test):
    """
    Use model to make prediction and save
    :param my_model:
    :param X_test:
    :param Y_test:
    :return:
    """

    predictions = my_model.predict(X_test)

    evaluation(predictions, Y_test)

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
all_subject_data = all_subject_data.query(f"Activity != {default_label}")

# Exlude a subject
# TODO: eval error should be on exluded subject
excluded_subject = all_subject_data.query(f"subject != \"{test_subject}\"")
# exlude_subject = all_subject_data.copy()

X_train, X_test, Y_train, Y_test = prepare_arrays(excluded_subject, test_size=test_size)

if train_model:
    eval_set = [(X_test, Y_test)]
    my_model = XGBClassifier(n_estimators=500, learning_rate=0.05, verbose=True)

    print("Training ...")
    # X_train = list(X_train)
    # Y_train = list(Y_train)
    # two = zip(X_train, Y_train)
    # for t in two:
    #     print(t)

    my_model.fit(X_train, Y_train, verbose=True, eval_set=eval_set, eval_metric="merror")
    results = my_model.evals_result()
    print(results)

    pickle.dump(my_model, open(save_model, "wb+"))
else:
    my_model = pickle.load(open(save_model, "rb"))

print("Feature importance:")
for fet in zip(selected_features, my_model.feature_importances_):
    print(f"{fet[0]}: {fet[1]}")

print("Testing on all subjects")
predict_activity(my_model, X_test, Y_test)

"""
Testing for particular subject
"""
print("-" * 20 + '\n', f"Performance on test subject {test_subject}:")
qeury = f"subject == \"{test_subject}\""
data = all_subject_data.query(qeury)
data = drop_data(data)

print("Number of test samples {0}".format(data.shape[0]))

X_train, X_test, Y_train, Y_test = prepare_arrays(data, test_size=test_size)
predictions = my_model.predict(X_test)

evaluation(predictions, Y_test)

pred = ["Prediction", "Actual", "Heart Rate"]
cleaned = pd.DataFrame(columns=pred)
cleaned["Prediction"] = list(predictions)
cleaned["Truth"] = list(Y_test)
cleaned["Heart Rate"] = [X[0] for X in X_test]

cleaned.to_csv('single_' + save_prediction)
