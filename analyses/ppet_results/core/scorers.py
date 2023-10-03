from typing import Sequence

import numpy as np

np.random.seed(0)

class Length(object):
    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates its total time range (max - min time)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation (not used in this function)
        :return: Time range for the patient, wrapped in a list
        """
        return [np.max(data[:, 1]) - np.min(data[:, 1])]

class Alerts(object):
    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates warnings per hour according to:
        (# of times the model output crosses the threshold) / (Time range of all data)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation
        :return: # of Warnings per hour, wrapped in a list
        """
        crossings = data[:, 2] >= threshold
        if any(crossings):
            return [np.sum(crossings) / (np.max(data[:, 1]) - np.min(data[:, 1]))]
        else:
            return [0]

class PosNeg(object):
    def __init__(self, tmin: float, tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """
        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates
        two booleans indicating if the model output did and did not cross the supplied threshold.

        Summing these values for case and control data will allow us to calculate the true and false test results.

        We consider data within a tmin and tmax as an extra measure to ensure that scores after the time of event
        are not included in the analysis. tmin=0 and tmax=np.inf will consider all data before the time of event.

        :param data: Matrix of current patient data Columns: (ID, Time, Model Output)
        :param threshold: Single threshold to be used in the score calculation
        :return: List of two booleans [Model output ever cross threshold?, Model output never cross threshold?]
        """

        # Find the candidate time points in the data
        if data == []:
            return [np.nan, np.nan]
        else:
            time_inds = np.logical_and(data[:, 1] >= self.tmin, data[:, 1] <= self.tmax)
        # Check that this patient has any data within tmin and tmax
        if any(time_inds):
            # Find the threshold crossings
            crossings = data[time_inds, 2] >= threshold
            # Determine if model output ever crossed the threshold
            pos = int(any(crossings))
            # Determine if model output never crossed the threshold
            neg = int(all(~crossings))
            return [pos, neg]
        # If no candidate time points exist return nan's
        else:
            return [np.nan, np.nan]

class ProportionWarning(object):
    def __init__(self, tmin: float, tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """

        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates warnings per hour according to:
        (# of times the model output crosses the threshold) / (Time range of all data)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation

        :return: Proportion of time spent under warning within tmin and tmax in a list
        """
        # Find candidate time indexes
        time_inds = np.logical_and(data[:, 1] >= self.tmin, data[:, 1] <= self.tmax)
        # Check if any candidate points exist
        if any(time_inds):
            # Make copy of the data to be safe
            data_time = data.copy()[time_inds, :]
            # Find crossings within th time window of interest
            crossings = data_time[:, 2] >= threshold
            return [np.sum(crossings) / np.sum(time_inds)]
        # If not candidates exist, return nan
        else:
            return [np.nan]
        
class ProportionWarning_case(object): # RanXiao, function added to handle False alarm proportion using whole duration omitting prediction horizon
    def __init__(self, tlead: float, twin: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """

        self.tlead = tlead
        self.twin = twin

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates warnings per hour according to:
        (# of times the model output crosses the threshold) / (Time range of all data)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation

        :return: Proportion of time spent under warning within tmin and tmax in a list
        """
        # RanXiao: Find candidate time indexes outside of prediction horizon (including both early and later alarms as false ones)
        time_inds = np.logical_or(data[:, 1] >= self.tlead+self.twin,data[:, 1] <= self.tlead)
        # Check if any candidate points exist
        if any(time_inds):
            # Make copy of the data to be safe
            data_time = data.copy()[time_inds, :]
            # Find crossings within th time window of interest
            crossings = data_time[:, 2] >= threshold
            return [np.sum(crossings) / np.sum(time_inds)]
        # If not candidates exist, return nan
        else:
            return [np.nan] 
        
class HourlyFalseAlarmRate(object):# RanXiao, function looks for number of false alarms/hour in control 
    def __init__(self, tmin: float, tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """

        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates warnings per hour according to:
        (# of times the model output crosses the threshold) / (Time range of all data)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation

        :return: Proportion of time spent under warning within tmin and tmax in a list
        """
        # Find candidate time indexes
        time_inds = np.logical_and(data[:, 1] >= self.tmin, data[:, 1] <= self.tmax)
        # Check if any candidate points exist
        if np.logical_and(any(time_inds),(max(data[:, 1])-min(data[:, 1]))>0):

            # Make copy of the data to be safe
            data_time = data.copy()[time_inds, :]
            # Find crossings within th time window of interest
            crossings = data_time[:, 2] >= threshold
            return [np.sum(crossings) / np.ceil(max(data[:, 1])-min(data[:, 1]))]
        # If not candidates exist, return nan
        else:
            return [np.nan]
        
class HourlyFalseAlarmRate_case(object): # RanXiao, function looks for number of false alarms/hour in case out of prediction horizon (too late + too early) 
    def __init__(self, tlead: float, twin: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """

        self.tlead = tlead
        self.twin = twin

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates warnings per hour according to:
        (# of times the model output crosses the threshold) / (Time range of all data)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation

        :return: Proportion of time spent under warning within tmin and tmax in a list
        """
        # RanXiao: Find candidate time indexes outside of prediction horizon (including both early and later alarms as false ones)
        time_inds = np.logical_or(data[:, 1] >= self.tlead+self.twin,data[:, 1] <= self.tlead)
        # Check if any candidate points exist
        if np.logical_and(any(time_inds),(max(data[:, 1])-min(data[:, 1])-self.twin)>0):
            # Make copy of the data to be safe
            data_time = data.copy()[time_inds, :]
            # Find crossings within th time window of interest
            crossings = data_time[:, 2] >= threshold
            return [np.sum(crossings) / np.ceil(max(data[:, 1])-min(data[:, 1])-self.twin)]
        # If not candidates exist, return nan
        else:
            return [np.nan]      

class HourlyFPR_case(object): # RanXiao, function added to handle HOURLY False alarm rate using whole duration omitting prediction horizon
    def __init__(self, tlead: float, twin: float,tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """

        self.tlead = tlead
        self.twin = twin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates warnings per hour according to:
        (# of times the model output crosses the threshold) / (Time range of all data)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation

        :return: Proportion of time spent under warning within tmin and tmax in a list
        """
        # RanXiao: Find candidate time indexes outside of prediction horizon (including both early and later alarms as false ones)
        time_inds = np.logical_or(np.logical_and(data[:, 1] >= self.tlead+self.twin, data[:, 1] <= self.tmax),data[:, 1] <= self.tlead)
        # Check if any candidate points exist
        if any(time_inds):
            # Make copy of the data to be safe
            data_time = data.copy()[time_inds, :]
            # Find crossings within th time window of interest
            crossings = data_time[:, 2] >= threshold   
            
            hourlyrate = []#Ran Xiao: this block calculate hourly FAR within in lead time
            hourly_start = 0
            hourly_end = 1
            while hourly_start<self.tlead:
                hourly_ind=np.logical_and(data_time[:, 1] >= hourly_start, data_time[:, 1] <= hourly_end)
                if any(hourly_ind):
                    hourlyrate.append(np.sum(crossings[hourly_ind]) / np.sum(hourly_ind))
                else:
                    hourlyrate.append(np.nan)    
                hourly_start+=1
                hourly_end+=1   
                    
            hourly_start = self.tlead+self.twin #Ran Xiao: this block calculate hourly FAR outside lead time + prediction horizon
            hourly_end = hourly_start+1
            while hourly_start<max(data_time[:,1]):
                hourly_ind=np.logical_and(data_time[:, 1] >= hourly_start, data_time[:, 1] <= hourly_end)
                if any(hourly_ind):
                    hourlyrate.append(np.sum(crossings[hourly_ind]) / np.sum(hourly_ind))
                else:
                    hourlyrate.append(np.nan)
                hourly_start+=1
                hourly_end+=1      
                 
            return [np.nanmean(hourlyrate)]
        # If not candidates exist, return nan
        else:
            return [np.nan]   
        
class HourlyFPR(object): # RanXiao, function added to handle HOURLY False alarm rate using whole duration for control data
    def __init__(self, tmin: float, tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """

        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates warnings per hour according to:
        (# of times the model output crosses the threshold) / (Time range of all data)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation

        :return: Proportion of time spent under warning within tmin and tmax in a list
        """
        # Find candidate time indexes
        time_inds = np.logical_and(data[:, 1] >= self.tmin, data[:, 1] <= self.tmax)
        # Check if any candidate points exist
        if any(time_inds):
            # Make copy of the data to be safe
            data_time = data.copy()[time_inds, :]
            # Find crossings within th time window of interest
            crossings = data_time[:, 2] >= threshold
            hourlyrate = []
            hourly_start = 0
            hourly_end = 1
            while hourly_start<max(data_time[:,1]):
                hourly_ind=np.logical_and(data_time[:, 1] >= hourly_start, data_time[:, 1] <= hourly_end)
                if any(hourly_ind):
                    hourlyrate.append(np.sum(crossings[hourly_ind]) / np.sum(hourly_ind))
                else:
                    hourlyrate.append(np.nan)
                hourly_start+=1
                hourly_end+=1                       
            return [np.nanmean(hourlyrate)]
        # If not candidates exist, return nan
        else:
            return [np.nan]   
        
class Profile(object):
    def __init__(self, tmin: float, tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """
        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates the number of early,
        on-time, and late warnings as well as the number of patients who were "missed" (no warnings
        generated before the time of event)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation
        :return: # of Warnings for each category [Early, On-Time, Late, Missed]
        """
        # Find data in the on-time range
        ontime_inds = np.logical_and(data[:, 1] < self.tmax, data[:, 1] >= self.tmin)
        # Find data in the early range
        early_inds = data[:, 1] >= self.tmax
        # Find data in the late range
        late_inds = np.logical_and(data[:, 1] < self.tmin, data[:, 1] >= 0)

        inds = [early_inds, ontime_inds, late_inds]
        temp_output = []
        for ind in inds:
            # Select data for the current time range
            cur_data = data[ind, :]
            # Calculate the number of threshold crossings
            cur_sum = np.sum(cur_data[:, 2] >= threshold)
            if cur_sum > 0:
                temp_output.append(cur_sum)
            else:
                temp_output.append(np.nan)
        # Add the number of patients who were "missed"
        temp_output.append(int(all(np.logical_and(data[:, 1] >= 0, data[:, 2] < threshold))))
        return temp_output

class ProfileNorm(object):
    def __init__(self, tmin: float, tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """
        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates the proportion of early,
        on-time, and late warnings that were "On" as well as the number of patients who were "missed" (no warnings
        generated before the time of event)

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation
        :return: List of lists containing [# "On" Warnings, # Warnings] for [Early, On-Time, Late, Missed]
        """
        # Find data in the on-time range
        ontime_inds = np.logical_and(data[:, 1] < self.tmax, data[:, 1] >= self.tmin)
        # Find data in the early range
        early_inds = data[:, 1] >= self.tmax
        # Find data in the late range
        late_inds = np.logical_and(data[:, 1] < self.tmin, data[:, 1] >= 0)

        inds = [early_inds, ontime_inds, late_inds]
        denominators = [np.sum(early_inds), np.sum(ontime_inds), np.sum(late_inds)]
        temp_output = []
        for denom, ind in zip(denominators, inds):
            # Select data for the current time range
            cur_data = data[ind, :]
            # Calculate the number of threshold crossings
            numerator = np.sum(cur_data[:, 2] >= threshold)
            if numerator > 0:
                temp_output.append([numerator, denom])
            else:
                temp_output.append([np.nan, np.nan])
        return temp_output

class TWarning(object):
    def __init__(self, tmin: float, tmax: float):
        """

        :param tmin: Minimum time (measured before the event or discharge) to filter out model outputs
        :param tmax: Maximum time (measured before the event or discharge) to filter out model outputs
        """
        self.tmin = tmin
        self.tmax = tmax

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates the first time a
        warning went "On" before the time of event

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation
        :return: Proportion of time spent under warning within tmin and tmax in a list
        """
        time_inds = np.logical_and(data[:, 1] >= self.tmin, data[:, 1] <= self.tmax)
        data_time = data[time_inds, :]
        data_thresh = data_time[data_time[:, 2] >= threshold, :]
        if data_thresh.shape[0] > 0:
            return np.max(data_thresh[:, 1])
        else:
            return np.nan

class Step(object):
    def __init__(self, max_time: float, step_size: float):
        """

        :param max_time: Maximum time to consider
        :param step_size: Step sizes to increment the analysis (starts at t=0)
        """
        self.max_time = max_time
        self.step_size = step_size

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates whether a
        warning was "On" less than T before the time of event where T is varied from
        0 to max_time by step_size.

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation
        :return: List of booleans indicating whether a warning was "On" for each time
        """
        t_list = []
        time = 0
        # Find all threshold crossings
        data_thresh = data[data[:, 2] >= threshold, :]
        while time < self.max_time:
            # Find threshold crossings less than "time" before the time of event
            inds = np.logical_and(data_thresh[:, 1] >= (time), data_thresh[:, 1] <= (time + self.step_size))
            # Store a boolean indicating if a warning was ever "On"
            t_list.append(any(inds))
            time += self.step_size
        return t_list

class PerHour(object):
    def __init__(self, max_time: float, step_size: float):
        """

        :param max_time: Maximum time to consider
        :param step_size: Step sizes to increment the analysis (starts at t=0)
        """
        self.max_time = max_time
        self.step_size = step_size

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates whether a
        warning was "On" within T0 and T0+step_size where T0 starts at 0 and is incremented by
        step_size until max_time is reached

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation
        :return: List of booleans indicating whether a warning was "On" within each time window
        """
        out = [];
        time = 0
        first = np.sum(np.logical_and(data[:, 1] >= 0, data[:, 1] <= (self.step_size)))
        while time <= self.max_time:
            # Find threshold crossings within time and time+step_size
            inds = np.logical_and(data[:, 1] > time, data[:, 1] <= (time + self.step_size))
            crossings = data[inds, 2] >= threshold
            test = any(inds)
            if test:
                if np.max(data[:, 1] <= time + 12):
                    out.append([int(any(crossings)), test, np.sum(inds) / first])
                else:
                    out.append([int(any(crossings)), test, np.nan])
            else:
                out.append([int(any(crossings)), test, np.nan])

            time += self.step_size
        return out

class Lead(object): #Ran, change prediction horizon of 12 to a variable as input twin
    def __init__(self, lead_times: Sequence[float],twin:float):
        """

        :param lead_times: Iterable of lead_times to consider
        """
        self.lead_times  = lead_times
        self.twin = twin

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates whether a
        warning was "On" within lead time and lead time + 12 hours

        :param data: Matrix of current patient data Columns: (ID, Time, Sc)
        :param threshold: Single threshold to be used in the score calculation
        :return: List of 2-length lists containing
        # of "On" warnings and # of warnings within the time range of interest
        """
        out = []
        first = np.sum(np.logical_and(data[:, 1] >= self.lead_times[0], data[:, 1] <= (self.lead_times[0] + self.twin)))
        for time in self.lead_times:
            inds = np.logical_and(data[:, 1] >= time, data[:, 1] <= (time + self.twin))
            crossings = data[inds, 2] >= threshold
            test = any(inds)
            if test:
                if np.max(data[:, 1] <= time + self.twin):
                    out.append([int(any(crossings)), test, np.sum(inds) / first])
                else:
                    out.append([int(any(crossings)), test, np.nan])
            else:
                out.append([int(any(crossings)), test, np.nan])
        return out

class SepsisUtility(object):
    def __init__(self, is_case: bool):
        """

        :param is_case:
        """
        self.is_case = is_case

    def __call__(self, data: np.ndarray, threshold: float):
        """
        This function takes data for a particular patient and calculates the physionet
        utility score from the sepsis challenge

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :param threshold: Single threshold to be used in the score calculation
        :return: Utility score
        """
        dt_early = 12
        dt_optimal = 6
        dt_late = 1
        max_u_tp = 1
        min_u_fn = -2
        u_fp = -0.05
        u_tn = 0

        def utility(t, prediction, is_case):
            # Define slopes and intercept points for utility functions of the form
            # u = m * t + b.
            m_1 = float(-max_u_tp) / float(dt_early - dt_optimal)
            b_1 = -m_1 * dt_early

            m_2 = float(max_u_tp) / float(dt_optimal - dt_late)
            b_2 = -m_2 * dt_late

            m_3 = float(-min_u_fn) / float(dt_optimal - dt_late)
            b_3 = -m_3 * dt_optimal
            # TP
            if is_case and prediction:
                if t >= dt_optimal:
                    return max(m_1 * t + b_1, u_fp)
                elif t >= dt_late:
                    return m_2 * t + b_2
                elif t <= dt_late:
                    return 0
            # FN
            elif is_case and not prediction:
                if t >= dt_optimal:
                    return 0
                elif t >= dt_late:
                    return m_3 * t + b_3
                elif t <= dt_late:
                    return min_u_fn

        if self.is_case:
            best_u = np.sum([utility(t, pred, True) for t, pred in
                             zip(data[:, 1], [False if t >= dt_early else True for t in data[:, 1]])])
            no_pred = np.sum([utility(t, pred, True) for t, pred in zip(data[:, 1], [False] * data.shape[0])])

            pos_inds = data[:, 2] >= threshold
            u = [utility(t, pred, self.is_case) for t, pred in zip(data[:, 1], pos_inds)]
            return [np.sum(u), best_u, no_pred]

        else:
            best_u = 0
            no_pred = 0
            pos_inds = data[:, 2] >= threshold
            return [u_fp * np.sum(pos_inds), best_u, no_pred]
