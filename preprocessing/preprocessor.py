import os
import pandas as pd
from dateutil import parser
import datetime

from cachetools.func import lru_cache

activities_meta_file = "Meta_data_activities.csv"
data_base_file = "/Bio-Monitor/Bio-M-Challenge-Data"


@lru_cache(maxsize=2000)
def parse_time(time):
    """
    To cache frequent parsing of time strings
    """

    return parser.parse(time)


def parse_epoch(int_time):
    fmt = "%H:%M:%S"

    # local time
    t = datetime.datetime.fromtimestamp(float(int_time) / 1000.)
    return t.strftime(fmt)


def get_biometrics_directory(base):
    """
    Returns the biometrics directories iterates, through all sujects, and days in subject
    :param subject_dir:
    :return:
        Dict with data on directories
    """

    directories = []

    # Gives each data dir id
    i = 0

    subject_dir = next(os.walk(base))[1]

    for subject in subject_dir:

        sub_dir = base + "/" + subject
        days = next(os.walk(sub_dir))[1]

        for day in days:
            # Variable numercial code for each biometric
            num = next(os.walk(sub_dir + "/" + day))[1][0]

            data = {
                "directory": sub_dir + "/" + day + "/" + num,
                "day": int(day[4]),
                "subject": subject[8],
                "number": i
            }

            directories.append(data)
            i += 1

    return directories


def format_biometric_file(file, biometric):
    return file + "/" + biometric + ".csv"


def clean_meta(meta_data):
    """
    Keeps required meta_data rows
    :param meta_data:
    :return:
    """

    # data = meta_data[meta_data.Position.notnull() and meta_data.Activity != "end"]
    data = meta_data.query('Activity.notnull() and Start_time.notnull()')

    return data


meta_data = pd.read_csv(activities_meta_file,
                        encoding='latin1')
meta_data = clean_meta(meta_data)


def get_range(meta_data, day, subject):
    """
    Yeild the start and end time of an activity
    Iterates through meta data activity
    :param meta_data: 
    :return: 
    """

    rows = meta_data.shape[0]

    for i in range(rows - 1):
        row = meta_data.iloc[[i]]

        if list(row["Day"])[0] == day and list(row["Subject"])[0] == subject:
            activity = list(row["Activity"])[0]

            time_start = list(row["Start_time"])[0]

            row = meta_data.iloc[[i + 1]]
            time_end = list(row["Start_time"])[0]

            yield time_start, time_end, activity


def clean_data(dir_dict):
    """
    Stiches together the biotric data from the selected files, and labels timestamped row with activity
    :param dir_dict:
    :return:
    """

    cleaned = pd.DataFrame(columns=biometics + columns)

    # Set the timestamps, assumed to be synced for all biometrics
    biometric_data = pd.read_csv(format_biometric_file(dir_dict["directory"], biometics[0]), encoding='latin1')
    # Get the timestamps column
    times = biometric_data[biometric_data.columns[0]]
    times = [parse_epoch(time) for time in times]
    cleaned["time"] = times

    cleaned["Activity"] = None
    cleaned["subject"] = dir_dict["subject"]
    cleaned["day"] = dir_dict["day"]

    for biometric in biometics:
        file = format_biometric_file(dir_dict["directory"], biometric)

        biometric_data = pd.read_csv(file, encoding='latin1')

        # Get the biometric column
        data = biometric_data[biometric_data.columns[1]]

        cleaned[biometric] = data

    for time_start, time_end, activity in get_range(meta_data, dir_dict["day"], dir_dict["subject"]):

        for i in cleaned.index:

            try:
                if parse_time(time_start) < parse_time(cleaned.at[i, 'time']) < parse_time(time_end):
                    cleaned.at[i, 'Activity'] = activity

            except (TypeError, ValueError) as e:
                print("Error Parsing Times: {0}".format(e))

    save_file = "Cleaned/sub_{0}_day{1}_id_{2}.csv".format(dir_dict["subject"], dir_dict["day"], dir_dict["number"])

    cleaned.to_csv(save_file, sep=',', encoding='utf-8')
    print("Done: {0}".format(dir_dict))


directories = get_biometrics_directory(data_base_file)

columns = ["Activity", "subject", "day"]
biometics = ["heart_rate", "tidal_volume_adjusted", "cadence", "step", "activity", "temperature_celcius",
             "systolic_pressure_adjusted", "minute_ventilation_adjusted"]

for dir_dict in directories:
    clean_data(dir_dict)
