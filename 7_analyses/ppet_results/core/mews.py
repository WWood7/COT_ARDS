import datetime
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from typing import Union, Dict, Callable

from core import utils

#path = '/home/alex/mews/data/'
path = '/Users/rx35/Documents/MATLAB/mews/'

def load():
    """
    Loads case and control data as well as information on event times (code blue)
    :return:
        - mews_code - MEWS data for case patients
        - mews_control - MEWS data for control patients
        - code - Information on event times for case patients
    """
    mews_case = pd.read_csv(path + 'mews_case.csv', low_memory=False)
    mews_control = pd.read_csv(path + 'mews_control.csv', low_memory=False)

    code = pd.read_csv(path + 'case_multiple.csv', sep=',', low_memory=False)

    return mews_case, mews_control, code


def create_case(code: pd.DataFrame, mews_case: pd.DataFrame, single_event: bool = False):
    """
    Processes raw data for case patients outputs a dictionary containing
    processed vital signs per patient encounter

    :param code: DataFrame of event times and encounter IDs
    :param mews_case: DataFrame of case patient vital signs
    :param single_event: Whether to consider patients with multiple events or not
    :return:
        - case_data - Dictionary (encounter ID's as keys) which contains vital sign values and times
                      (T=0 corresponds to time of event)
    """
    encounters = np.array(mews_case['encounter_ID'].astype(int))
    dates = mews_case['FlowDate']
    times = mews_case['FlowTime']
    names = mews_case['Name']
    values = np.array(mews_case['Value'].astype(float))

    # Use the previously determined indexes to access the values of interest
    code_times = code['CodeTime']
    code_encounters = np.array(code['Encounter_ID'].astype(int))

    # Find unique IDS
    code_encounters_unique, counts = np.unique(code_encounters, return_counts=True)

    # Find the IDS that occur only once
    if single_event:
        code_encounters_unique_multiple = code_encounters_unique[np.where(counts == 1)[0]]
    else:
        code_encounters_unique_multiple = code_encounters_unique[np.where(counts > 1)[0]]

    case_data = dict()
    # Iterate through list of filtered encounters
    for encounter in code_encounters_unique[np.isin(code_encounters_unique, np.unique(encounters))]:
        # Find the index in the code and flowsheet files, corresponding to the current encounter
        formatted_code_time = np.min([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in
                                      code_times[np.where(code_encounters == encounter)[0]]])
        encounter_inds = np.where(encounters == encounter)[0]

        # Convert the time stamps for all the recorded values into a list of single strings
        date_times = [date + ' ' + time for date, time in zip(dates[encounter_inds], times[encounter_inds])]

        # Convert the dates/times from strings to datetime objects
        formatted_date_times = np.asarray(
            [datetime.datetime.strptime(item, '%Y-%m-%d %H:%M:%S.0000000') for item in date_times])

        # Subtract the minimum datetime value from all datetime objects
        formatted_code_time -= np.min(formatted_date_times)

        # Take the code time and re-reference according to the same minimum datetime value
        formatted_date_times = formatted_date_times - np.min(formatted_date_times)

        # Convert the offset datetime differences into seconds
        seconds_date_times = np.asarray([item.total_seconds() for item in formatted_date_times]).astype(int)

        if np.isin(encounter, code_encounters_unique_multiple):
            seconds_date_times = seconds_date_times[seconds_date_times <= formatted_code_time.total_seconds()]

        # Initialize data array for current encounter
        def empty():
            temp = np.empty((len(np.unique(seconds_date_times)),))
            temp[:] = np.nan
            return temp

        encounter_data = defaultdict(empty)
        cur_times = np.unique(seconds_date_times)

        # Iterate through the recorded values and assign them to the recorded time index
        # If a certain feature/signal was not recorded at a certain time point, assign Nan
        for name, value, time in zip(names[encounter_inds], values[encounter_inds], seconds_date_times):
            cur_ind = np.where(cur_times == time)[0][0]
            encounter_data[name][cur_ind] = value

        # Store current data into a dictionary with descriptive keys
        # time values, data values
        data = dict()
        data['data'] = encounter_data
        data['time'] = (cur_times - formatted_code_time.total_seconds()) / 3600
        case_data[int(encounter)] = data

    return case_data


def create_control(mews_control: pd.DataFrame):
    """
    Processes raw data for control patients outputs a dictionary containing
    processed vital signs per patient encounter
    :param mews_control: DataFrame of control patient vital signs
    :return:
        - control_data - Dictionary (encounter ID's as keys) which contains vital sign values and times
                        (T=0 corresponds to time of discharge)
    """
    encounters = np.array(mews_control['encounter_ID'].astype(int))
    dates = mews_control['FlowDate']
    times = mews_control['FlowTime']
    names = mews_control['Name']
    values = mews_control['Value']

    bp = np.where(mews_control['Name'] == 'BLOOD PRESSURE')[0]
    bp_split = bp[[i for i, x in enumerate(values[bp]) if type(x) != float]]
    # Control data had systolic and diastolic BP. Need to filter out SBP
    values[bp_split] = [int(x.split('/')[0]) for x in values[bp_split]]
    names[bp] = 'BLOOD PRESSURE SYSTOLIC'

    control_data = {}
    # Iterate through list of filtered encounters
    for encounter in np.unique(encounters):
        # Find the index in the code and flowsheet files, corresponding to the current encounter
        encounter_inds = np.where(encounters == encounter)[0]

        # Convert the time stamps for all the recorded values into a list of single strings
        date_times = [date + ' ' + time.split('.')[0] for date, time in
                      zip(dates[encounter_inds], times[encounter_inds])]

        # Convert the dates/times from strings to datetime objects
        formatted_date_times = np.asarray(
            [datetime.datetime.strptime(item, '%Y-%m-%d %H:%M:%S') for item in date_times])
        formatted_date_times = formatted_date_times - np.max(formatted_date_times)

        # Convert the offset datetime differences into seconds
        seconds_date_times = np.asarray([item.total_seconds() for item in formatted_date_times]).astype(int)

        # Initialize data array for current encounter
        def empty():
            temp = np.empty((len(np.unique(seconds_date_times)),))
            temp[:] = np.nan
            return temp

        encounter_data = defaultdict(empty)
        cur_times = np.unique(seconds_date_times)

        # Iterate through the recorded values and assign them to the recorded time index
        # If a certain feature/signal was not recorded at a certain time point, assign Nan
        for name, value, time in zip(names[encounter_inds], values[encounter_inds], seconds_date_times):
            cur_ind = np.where(cur_times == time)[0][0]
            encounter_data[name][cur_ind] = value

        # Store current data into a dictionary with descriptive keys
        # time values, data values, and code time
        data = {}
        data['data'] = encounter_data
        data['time'] = (cur_times - np.max(cur_times)) / 3600
        control_data[int(encounter)] = data

    return control_data


def mews_persist(data, times, period=False, scorer=False):
    missing_lengths = defaultdict(lambda: 0)
    bad_inds = []
    # For each vital sign
    for k in ['BLOOD PRESSURE SYSTOLIC', 'PULSE', 'RESPIRATIONS', 'TEMPERATURE', 'R CPN GLASGOW COMA SCALE SCORE']:
        # Get values for current vital sign
        cur_data = data[k]
        # Find the missing and real time points for the vital sign
        missing_inds = np.isnan(cur_data)
        real_inds = np.where(~missing_inds)[0]
        # Aux calculation, calculate the number of values that are missing
        missing_lengths[k] = np.sum(missing_inds)

        real_times = times[real_inds]
        real_data = cur_data[real_inds]

        # Check if there is any data (real not missing) for a particular vital sign
        if len(real_inds) == 0:
            # If its GCS, move on
            if k == 'R CPN GLASGOW COMA SCALE SCORE':
                pass
            # Else return a null response
            else:
                return [], [], defaultdict(lambda: 0), 0, np.arange(len(times))

        # Check if there is any data (real not missing) before the time of event
        else:
            first_time = np.min(real_times)
            if first_time > 0:
                # If GCS, move on
                if k == 'R CPN GLASGOW COMA SCALE SCORE':
                    pass
                # Else return null response
                else:
                    return [], [], defaultdict(lambda: 0), 0, np.arange(len(times))

            # Check if the first real value was after the time of event
            if first_time > np.min(times):
                # Add the non-real time points before the time of event to the list of bad indices
                bad_inds += np.where(times < first_time)[0].tolist()

            # If only one real value exists, impute it forward for all other time points
            if len(real_times) == 1:
                cur_data[:] = cur_data[real_inds[0]]

            # Else actually call interpolation to fill the gaps
            else:
                f = interp1d(real_times, real_data, bounds_error=False, kind='previous', fill_value='extrapolate')
                cur_data[:] = f(times)

    # Single index could be bad for several reasons. Get a unique list of bad indices
    bad_inds = np.unique(bad_inds)

    # If all the recorded values are bad, return null response
    if len(bad_inds) == len(times):
        print('No good inds')
        return [], [], defaultdict(lambda: 0), 0, bad_inds

    # Delete all the bad values for each vital sign
    for k in data.keys():
        data[k] = np.delete(data[k].copy(), bad_inds)

    # Delete all the bad values for the times
    times = np.delete(times.copy(), bad_inds)

    # Capture the case where GCS was not recorded
    # If so, clip the NaN's to be same length as the other vital signs
    if len(data['R CPN GLASGOW COMA SCALE SCORE']) != len(times):
        data['R CPN GLASGOW COMA SCALE SCORE'] = data['R CPN GLASGOW COMA SCALE SCORE'][:len(times)]

    sbp = data['BLOOD PRESSURE SYSTOLIC']
    hr = data['PULSE']
    coma = data['R CPN GLASGOW COMA SCALE SCORE']
    resp = data['RESPIRATIONS']
    temp = data['TEMPERATURE']

    # If performing regular calculation at the vital sign level
    if period and scorer:
        start = np.min(times)
        start_ind = np.argmin(times)

        # Want to calculate a score based on first available vital signs
        # Then apply fixed interval calculation from here
        sbp_new = [sbp[start_ind]]
        hr_new = [hr[start_ind]]
        coma_new = [coma[start_ind]]
        resp_new = [resp[start_ind]]
        temp_new = [temp[start_ind]]
        times_new = [start]

        while start + period < 0:
            cur_inds = np.where((times >= start) & (times <= (start + period)) & (times <= 0))[0]
            if len(cur_inds) > 0:
                # Check if the function takes into account the type of vital sign
                # The function must have "vital_name" as an argument
                if 'vital_name' in scorer.__code__.co_varnames:
                    sbp_new.append(scorer(sbp[cur_inds], 'sbp'))
                    hr_new.append(scorer(hr[cur_inds], 'hr'))
                    coma_new.append(scorer(coma[cur_inds], 'coma'))
                    resp_new.append(scorer(resp[cur_inds], 'resp'))
                    temp_new.append(scorer(temp[cur_inds], 'temp'))

                # Else just naively apply the function
                else:
                    sbp_new.append(scorer(sbp[cur_inds]))
                    hr_new.append(scorer(hr[cur_inds]))
                    coma_new.append(scorer(coma[cur_inds]))
                    resp_new.append(scorer(resp[cur_inds]))
                    temp_new.append(scorer(temp[cur_inds]))
                times_new.append(start + period)
            start += period

        sbp = np.array(sbp_new)
        hr = np.array(hr_new)
        coma = np.array(coma_new)
        resp = np.array(resp_new)
        temp = np.array(temp_new)

        # Make sure to mutate the original data with the newly calculated vital signs
        data['BLOOD PRESSURE SYSTOLIC'] = sbp
        data['PULSE'] = hr
        data['R CPN GLASGOW COMA SCALE SCORE'] = coma
        data['RESPIRATIONS'] = resp
        data['TEMPERATURE'] = temp
    else:
        times_new = times

    scores = np.zeros((len(sbp),))
    # Create lists of time indexes for each particular MEWS value (0, 1, 2, 3)
    indexes = [[], [], [], []]

    # Calculate MEWS for SBP
    indexes[3].append(np.where(sbp <= 70)[0])
    indexes[2].append(np.where(np.logical_and(sbp >= 71, sbp <= 80))[0])
    indexes[1].append(np.where(np.logical_and(sbp >= 81, sbp <= 100))[0])
    indexes[0].append(np.where(np.logical_and(sbp >= 101, sbp <= 199))[0])
    indexes[3].append(np.where(sbp >= 200)[0])

    # Calculate MEWS for heart rate
    indexes[2].append(np.where(hr <= 40)[0])
    indexes[1].append(np.where(np.logical_and(hr >= 41, hr <= 50))[0])
    indexes[0].append(np.where(np.logical_and(hr >= 51, hr <= 100))[0])
    indexes[1].append(np.where(np.logical_and(hr >= 101, hr <= 110))[0])
    indexes[2].append(np.where(np.logical_and(hr >= 111, hr <= 129))[0])
    indexes[3].append(np.where(hr >= 130)[0])

    # Calculate MEWS for respiratory rate
    indexes[2].append(np.where(resp < 9)[0])
    indexes[0].append(np.where(np.logical_and(resp >= 9, resp <= 14))[0])
    indexes[1].append(np.where(np.logical_and(resp >= 15, resp <= 20))[0])
    indexes[2].append(np.where(np.logical_and(resp >= 21, resp <= 29))[0])
    indexes[3].append(np.where(resp >= 30)[0])

    # Calculate MEWS for temperature
    indexes[2].append(np.where(temp < 95)[0])
    indexes[0].append(np.where(np.logical_and(temp >= 95, temp <= 101.12))[0])
    indexes[2].append(np.where(temp >= 101.13)[0])

    # Calculate MEWS for GCS score
    indexes[0].append(np.where(np.logical_and(coma >= 14, coma <= 15))[0])
    indexes[1].append(np.where(np.logical_and(coma >= 10, coma <= 13))[0])
    indexes[2].append(np.where(np.logical_and(coma >= 4, coma <= 9))[0])
    indexes[3].append(np.where(coma == 3)[0])

    # Total up scores at each time point
    for i, ind in enumerate(indexes):
        for cur_ind in ind:
            for id in cur_ind:
                scores[id] += i

    return scores, np.array(times_new), missing_lengths, len(times), bad_inds


def calculate_scores(data: Dict[int, Dict[str, np.ndarray]], period: float, scorer: Callable, data_level: bool = False):
    """

    :param data:
    :param period:
    :param scorer:
    :param data_level:
    :return:
    """
    missing = defaultdict(lambda: 0)
    ranges = defaultdict(lambda: 0)
    numbers = defaultdict(lambda: 0)

    total = 0
    bad_encounters = []
    for i, (k, v) in enumerate(data.items()):
        v['regular_scores'] = []
        v['regular_times'] = []

        for key in v['data'].keys():
            ranges[key] += np.abs(np.min(v['time']))
            numbers[key] += np.sum(~np.isnan(v['data'][key]))

        if data_level:
            v['scores'], v['time'], cur_missing, cur_total, bad_inds = mews_persist(v['data'], v['time'], period,
                                                                                    scorer)
        else:
            v['scores'], v['time'], cur_missing, cur_total, bad_inds = mews_persist(v['data'], v['time'])

        v['raw_time'] = v['time'].copy()

        for key in cur_missing.keys():
            missing[key] += cur_missing[key]
        total += cur_total

        # If null response detected, add this encounter to list of bad encounters
        if v['scores'] == []:
            bad_encounters.append(k)
            continue
        else:
            # Double check that there is some real data before the time of event
            if np.sum(v['time'] <= 0) == 0:
                bad_encounters.append(k)

        # If performing regular MEWS calculation at the score level
        if not data_level and period:
            times = v['time']
            scores = v['scores']
            start_ = times[0]

            while start_ + period < 0:
                def regular(start):
                    cur_inds = np.where(np.logical_and(times >= start, times <= (start + period)))[0]
                    if len(cur_inds) > 0:
                        cur_scores = scores[cur_inds]
                        v['regular_times'].append(start + period)
                        v['regular_scores'].append(scorer(cur_scores))
                    else:
                        v['regular_times'].append(start + period)
                        v['regular_scores'].append(v['regular_scores'][-1])

                regular(start_)
                start_ += period

            v['scores'] = v['regular_scores']
            v['time'] = v['regular_times']
        v['data'] = dict(v['data'])

    # Delete all the bad encounters
    for k in bad_encounters:
        del data[k]

    print("Proportion of values imputed:")
    for k in missing.keys():
        print(k, missing[k] / (numbers[k] + missing[k]))

    print("Mean sampling frequencies:")
    for k in missing.keys():
        print(k, numbers[k] / ranges[k])


def prepare(data):
    '''
    Takes data as a dictionary. Each key is an encounter id and the values are
    dictionaries containing the time and value of each MEWS score, as well as the data used to compute the scores.

    All encounter data is concatenated into one numpy array.

    This was done because serializing dictionaries in python is difficult and consumes too much RAM.

    :param data: dictionary of data. Key = encounter ID. Value = dictionary with keys: time, scores, and data
    :return:
    '''
    times = []
    scores = []
    e = []
    for encounter in data.keys():
        times.append(np.array(data[encounter]['time']))
        scores.append(np.array(data[encounter]['scores']))
        e.append(np.array([encounter] * len(times[-1])))
    return np.hstack((np.concatenate(e).reshape(-1, 1), -np.concatenate(times).reshape(-1, 1),
                      np.concatenate(scores).reshape(-1, 1)))


def prepare_case_multiple(period=False, scorer=utils.base, data_level=False):
    ''''
    Checks if the data has been prepared for case patients with multiple events.

    If not, it prepares the data by calling the appropriate functions

    :return:
        - prepared_case - numpy array of
    '''
    name = scorer.__name__

    if name == 'base':
        per = ''
        period = False
    else:
        per = period

    type = '_data' * data_level

    if os.path.isfile(f'{path}prepared_case_multiple_{name}_{per}{type}.pkl'):
        prepared_case = pickle.load(open(f'{path}prepared_case_multiple_{name}_{per}{type}.pkl', 'rb'))
        case = pickle.load(open(f'{path}/prepared_case_multiple_{name}_{per}{type}_raw.pkl', 'rb'))
        return prepared_case, case

    else:
        # Load the data
        mews_code, mews_control, code = load()
        # Process the data
        case = create_case(code, mews_code, False)
        # Calculate MEWS scores
        calculate_scores(case, period, scorer, data_level)
        # Prepare as a single numpy array
        prepared_case = prepare(case)
        with open(f'{path}prepared_case_multiple_{name}_{per}{type}.pkl', 'wb', ) as f:
            pickle.dump(prepared_case, f)
        with open(f'{path}prepared_case_multiple_{name}_{per}{type}_raw.pkl', 'wb', ) as f:
            pickle.dump(dict(case), f)

        return prepared_case, case


def prepare_control(period: Union[bool, float] = False, scorer: Callable = utils.base, data_level: bool = False):
    """
    Checks if the data has bee prepared for control patients

    If not, it prepares the data by calling the appropriate functions

    :param period:
    :param scorer:
    :param data_level:
    :return:
    """
    name = scorer.__name__

    if name == 'base':
        per = ''
        period = False
    else:
        per = period

    type = '_data' * data_level

    if os.path.isfile(f'{path}prepared_control_{name}_{per}{type}.pkl'):
        prepared_control = pickle.load(open(f'{path}prepared_control_{name}_{per}{type}.pkl', 'rb'))
        control = pickle.load(open(f'{path}prepared_control_{name}_{per}{type}_raw.pkl', 'rb'))
        return prepared_control, control
    else:
        # Load the data
        mews_code, mews_control, code = load()
        # Process the data
        control = create_control(mews_control)
        # Calculate mews scores (no need for reassignment, dictionaries are mutable)
        calculate_scores(control, period, scorer, data_level)
        # Prepare the data as a single numpy array
        prepared_control = prepare(control)
        with open(f'{path}prepared_control_{name}_{per}{type}.pkl', 'wb', ) as f:
            pickle.dump(prepared_control, f)
        with open(f'{path}prepared_control_{name}_{per}{type}_raw.pkl', 'wb', ) as f:
            pickle.dump(control, f)
        return prepared_control, control
