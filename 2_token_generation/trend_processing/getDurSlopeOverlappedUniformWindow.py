import numpy as np
import scipy.stats as st

def getDurSlopeOverlappedUniformWindow(win_t, x, t, increment_t):
    #This function uses a robust linear regression on each of the overlapped
    #windows and calculates slope, intercept, start time and end time of each
    #window.
    #Input: win_t - window length converted to a numeric value in Matlab
    #       x - values of the time series at time specified by t
    #       t - timestamps converted to numeric values in Matlab (see datenum)
    #       increment_t - window increment converted to a numeric value in Matlab
    #Output: dur_slope - slope of all the overlapped windows
    #        dur_intercept - intercept of all the overlapped windows
    #        dur_time_start - start time of all the overlapped windows (numeric)
    #        dur_time_end - end time of all the overlapped windows (numeric)

    dur_time_start = list(range(t[0],t[-1]-win_t+1))
    dur_time_end = [val+win_t for val in dur_time_start]

    dur_slope = np.full((1,len(dur_time_start)), np.nan)
    dur_intercept = np.full((1,len(dur_time_start)), np.nan)

    for i in range(len(dur_time_start)):
        Idx = t[(t >= dur_time_start[i]) & (t <= dur_time_end[i])]
        #print(Idx)
        tmp_t = t[Idx]
        tmp_x = x[Idx]

        if len(tmp_x[~np.isnan(tmp_x)]) >= 3:
            # robust linear regression
            #b = robustfit(tmp_t, tmp_x)
            slope, intercept, r_value, p_value, std_err = st.linregress(tmp_t, tmp_x)
            #print(slope)
            dur_slope[0,i] = slope
            #print(dur_slope)
            dur_intercept[0,i] = intercept

    return dur_slope, dur_intercept, dur_time_start, dur_time_end