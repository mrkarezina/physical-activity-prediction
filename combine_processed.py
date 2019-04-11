import pandas as pd
import datetime

import os

cleaned_data_dir = "/Cleaned"
combined_data_csv = "combined_biometric.csv"


def combine_csv(base_dir):
    """
    Combine all csv in base dir
    :param base_dir:
    :return:
    """

    csvs = next(os.walk(base_dir))[2]

    combined = None
    for csv in csvs:
        if combined is None:
            combined = pd.read_csv(base_dir + "/" + csv, low_memory=False)
            combined = combined.query('Activity.notnull() and not (Activity == "end")')
        else:
            new = pd.read_csv(base_dir + "/" + csv, low_memory=False)
            new = new.query('Activity.notnull() and not (Activity == "end")')
            combined = combined.append(new, ignore_index=True)

    combined = combined.drop("Unnamed: 0", axis=1)
    combined.to_csv(combined_data_csv, sep=',', encoding='utf-8')


def create_series(data):
    """
    Takes the pandas dataframe and sets an activity_session id to each row
    Activity session a successive series of activities
    :param data:
    :return:
    """

    cleaned = data.copy()
    cleaned["session"] = None

    # The permitted amount differnce time for rows to be same activity
    same_session_threshold = datetime.timedelta(minutes=15)

    activity_sessions = 0

    # Query for each person
    subjects = pd.unique(data["subject"])
    days = pd.unique(data["day"])

    for subject in subjects:
        for day in days:
            subject_data = data.query('subject == \'{0}\' and day == {1}'.format(subject, day))

            subject_data['time'] = pd.to_datetime(subject_data.time, format="%H:%M:%S")
            subject_data.sort_values(by='time')

            subject_data = subject_data.reset_index()

            try:
                previous_time = subject_data.at[0, 'time']
                for row in range(1, subject_data.shape[0] - 1):

                    if (subject_data.at[row, 'time'] - previous_time) > same_session_threshold:
                        activity_sessions += 1

                    cleaned.at[row, "session"] = activity_sessions
            except KeyError:
                print("Key error day")

    cleaned.to_csv('series_bio.csv', sep=',', encoding='utf-8')


combine_csv(cleaned_data_dir)

data = pd.read_csv()
create_series(data)
