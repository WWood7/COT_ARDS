import numpy as np
from trend_processing.getDurSlopeOverlappedUniformWindow import getDurSlopeOverlappedUniformWindow
from trend_processing.DT_Sign import DT_Sign
import scipy.stats as st

def timedelta_to_hour(time):
    d = 24*(time.days)
    h = (time.seconds)/3600
    total_hours = d + h
    return total_hours

def DominantTrendDetection(x, t):
    ###This function detects the dominant trend from a time series.
    #Inputs: x -- values of the time series at time specified by t
    #       t -- timestamps converted to numeric values in Matlab (see datenum)
    #
    #Outputs: Duration -- duration of the dominant trend
    #        DT_start_t -- start time of the dominant trend (numeric value)
    #        DT_end_t -- end time of the dominant trend (numeric value)
    #        OWL -- optimal window length


    #In Matlab, if x is a datenum variable, then x + 1 is one day after x.
    #Hour = 1/24;

    if len(x[~ np.isnan(x)])==0: #if x is all nan
        Duration=0
        DT_start_t=np.nan
        DT_end_t=np.nan
        OWL=np.nan
        return Duration,DT_start_t,DT_end_t, OWL

    #Search over various window length
    winL = range(3,6) #hours
    increment = 1 #hour

    #Convert window length in minutes to Matlab unit
    win_t = winL
    increment_t = increment

    #Percentage of negative and positive slope signs
    percentageNeg = np.zeros(len(win_t))
    percentagePos = np.zeros(len(win_t))
    for i in range(len(win_t)):
        #print(win_t[i])
        dur_slope, dur_intercept, dur_time_start, dur_time_end = getDurSlopeOverlappedUniformWindow(win_t[i], x, t, increment_t)
        #print(dur_slope)
        percentageNeg[i] = len(dur_slope[dur_slope<0])/len(dur_slope[~ np.isnan(dur_slope)])
        percentagePos[i] = len(dur_slope[dur_slope>0])/len(dur_slope[~ np.isnan(dur_slope)])

    #negative slopes
    Max_Percent_Neg = max(abs(percentageNeg))
    Idx_Neg=list(abs(percentageNeg)).index(Max_Percent_Neg)
    Sign=-1 #%-
    winL_Neg=winL[Idx_Neg] #hours
    win_t=winL_Neg
    #While we do not use a filter here, a Hamepl filter can help when there are
    #outliers in the time series.
    x_filt=x
    dur_slope, dur_intercept, dur_time_start, dur_time_end = getDurSlopeOverlappedUniformWindow(win_t, x_filt, t, increment_t)
    #Dominant decreasing trend
    #print('!!!!!!!!!!', Sign, Max_Percent_Neg, win_t, len(dur_slope[0,:]))
    DT_start, DT_L=DT_Sign(dur_slope[0,:], Sign)
    if np.isnan(DT_start) or DT_L==0:
        DT_start_t_Neg=np.nan
        DT_end_t_Neg=np.nan
        Duration_Neg=0
        DT_slope_Neg=np.nan
        DT_terminal_Neg=x[-1]
    else:
        DT_start_t_Neg=dur_time_start[DT_start]
        DT_end_t_Neg=dur_time_end[DT_start+DT_L-1]
        Duration_Neg=DT_end_t_Neg-DT_start_t_Neg
        DT_Idx_Neg = t[(t >= DT_start_t_Neg) & (t <= DT_end_t_Neg)]
        # print(Idx)
        tmp_t_Neg = t[DT_Idx_Neg]
        tmp_x_Neg = x[DT_Idx_Neg]
        DT_slope_Neg, DT_intercept_Neg, _, _, _ = st.linregress(tmp_t_Neg, tmp_x_Neg)
        DT_terminal_Neg = DT_intercept_Neg + Duration_Neg*DT_slope_Neg

    #postive slopes
    Max_Percent_Pos=max(abs(percentagePos))
    Idx_Pos=list(abs(percentagePos)).index(Max_Percent_Pos)
    Sign=1 #+
    winL_Pos=winL[Idx_Pos] #hours
    win_t=winL_Pos
    #While we do not use a filter here, a Hamepl filter can help when there are
    #outliers in the time series.
    x_filt=x
    dur_slope, dur_intercept, dur_time_start, dur_time_end = getDurSlopeOverlappedUniformWindow(win_t, x_filt, t, increment_t)
    #Dominant increasing trend
    #print(Sign, Max_Percent_Pos, win_t, len(dur_slope[0,:]))
    DT_start, DT_L=DT_Sign(dur_slope[0,:], Sign)
    if np.isnan(DT_start) or DT_L==0:
        DT_start_t_Pos=np.nan
        DT_end_t_Pos=np.nan
        Duration_Pos=0
        DT_slope_Pos= np.nan
        DT_terminal_Pos = x[-1]
    else:
        DT_start_t_Pos=dur_time_start[DT_start]
        DT_end_t_Pos=dur_time_end[DT_start+DT_L-1]
        Duration_Pos=DT_end_t_Pos-DT_start_t_Pos
        DT_Idx_Pos = t[(t >= DT_start_t_Pos) & (t <= DT_end_t_Pos)]
        # print(Idx)
        tmp_t_Pos = t[DT_Idx_Pos]
        tmp_x_Pos = x[DT_Idx_Pos]
        DT_slope_Pos, DT_intercept_Pos, _, _, _ = st.linregress(tmp_t_Pos, tmp_x_Pos)
        DT_terminal_Pos = DT_intercept_Pos + Duration_Pos * DT_slope_Pos

    #Compare dominant decreasing and increasing trends
    if Duration_Neg==0 or Duration_Pos>Duration_Neg:
        Duration=Duration_Pos
        DT_start_t=DT_start_t_Pos
        DT_end_t=DT_end_t_Pos
        OWL=winL_Pos
        DT_sign = 1
        DT_slope = DT_slope_Pos
        DT_terminal = DT_terminal_Pos
    else:
        Duration=Duration_Neg
        DT_start_t=DT_start_t_Neg
        DT_end_t=DT_end_t_Neg
        OWL=winL_Neg
        DT_sign = -1
        DT_slope = DT_slope_Neg
        DT_terminal = DT_terminal_Neg

    return Duration,DT_start_t,DT_end_t, OWL, DT_sign, DT_slope, DT_terminal

def L1DominantTrendDetection(slope, slope_duration, kink_value):
    dur_sign = np.sign(slope).tolist()

    ## concat all consecutive pos
    DT_start_pos = np.nan
    DT_L_pos = 0
    DT_termianl_pos = kink_value[-1]
    DT_slope_pos = np.nan

    # Current window
    CW_start_pos = np.nan
    CW_L_pos = 0
    # Those in dur_slope whose sign matches 'Sign' are converted to 1.
    # nan and not matching 'Sign' are converted to 0.
    # Then the problem becomes finding the longest 1's.
    Idx_pos = [1 if val == 1 else 0 for val in dur_sign]
    # print(Idx)
    previous = 0
    for i in range(len(Idx_pos)):
        if Idx_pos[i] == 0:  # if Idx(i) is 0
            previous = 0
        else:  # if Idx(i) is 1
            if previous == 0:  # start of 1's
                previous = 1
                CW_start_pos = i
                CW_L_pos = slope_duration[i]
            else:  # in the middle of 1's
                previous = 1
                CW_L_pos = CW_L_pos + slope_duration[i]

            if CW_L_pos > DT_L_pos:
                DT_start_pos = CW_start_pos
                DT_L_pos = CW_L_pos
                DT_termianl_pos = kink_value[i+1]
                DT_slope_pos = (kink_value[i+1] - kink_value[i])/DT_L_pos


    ## concat all consecutive pos
    DT_start_neg = np.nan
    DT_L_neg = 0
    DT_termianl_neg = kink_value[-1]
    DT_slope_neg = np.nan

    # Current window
    CW_start_neg = np.nan
    CW_L_neg = 0
    # Those in dur_slope whose sign matches 'Sign' are converted to 1.
    # nan and not matching 'Sign' are converted to 0.
    # Then the problem becomes finding the longest 1's.
    Idx_neg = [1 if val == -1 else 0 for val in dur_sign]
    # print(Idx)
    previous = 0
    for i in range(len(Idx_neg)):
        if Idx_neg[i] == 0:  # if Idx(i) is 0
            previous = 0
        else:  # if Idx(i) is 1
            if previous == 0:  # start of 1's
                previous = 1
                CW_start_neg = i
                CW_L_neg = slope_duration[i]
            else:  # in the middle of 1's
                previous = 1
                CW_L_neg = CW_L_neg + slope_duration[i]

            if CW_L_neg > DT_L_neg:
                DT_start_neg = CW_start_neg
                DT_L_neg = CW_L_neg
                DT_termianl_neg = kink_value[i + 1]
                DT_slope_neg = (kink_value[i + 1] - kink_value[i]) / DT_L_neg

    if DT_L_neg>DT_L_pos:
        DT_Duration = DT_L_neg
        DT_sign = -1
        DT_slope = DT_slope_neg
        DT_terminal = DT_termianl_neg
    else:
        DT_Duration = DT_L_pos
        DT_sign = 1
        DT_slope = DT_slope_pos
        DT_terminal = DT_termianl_pos
    return DT_Duration, DT_sign, DT_slope, DT_terminal
'''
def get_dt_l1_plot(imputed_vital_ards, train_ards_hadm_onset, vital_name, duration, delta_value):
    train_ards_hadm_onset = train_ards_hadm_onset.rename(columns={"charttime": "onsettime"})
    vital_ards = pd.merge(
        imputed_vital_ards[['hadm_id', 'charttime', vital_name]], train_ards_hadm_onset[['hadm_id', 'onsettime']],
        how="inner", on=["hadm_id"])
    print(len(imputed_vital_ards), len(train_ards_hadm_onset), len(vital_ards['hadm_id'].unique()))
    print(vital_ards.head())
    print(vital_ards[vital_name].mean(), vital_ards[vital_name].median(), vital_ards[vital_name].max(),
          vital_ards[vital_name].min())

    uniperIDs = vital_ards['hadm_id'].unique()
    uniperIDs.sort()
    dataHadmID = []
    for id in range(len(uniperIDs)):
        dataHadmID.append(vital_ards[vital_ards['hadm_id'] == uniperIDs[id]])
    for i in range(len(dataHadmID)):
        dataHadmID[i] = dataHadmID[i].sort_values(by=['charttime'])
    print(len(dataHadmID), 'encounters')

    vital_start = []
    vital_end = []
    count_total_before = 0
    count_half_before = 0
    count_middle = 0
    count_half_after = 0
    count_total_after = 0
    count_cross = 0
    for p in dataHadmID:
        p = p.reset_index(drop=True)
        hadm_id = p['hadm_id'][0]
        p['charttime'] = pd.to_datetime(p['charttime'])
        p['onsettime'] = pd.to_datetime(p['onsettime'])
        p['interval'] = p['onsettime'] - p['charttime']
        temp = [abs(timedelta_to_hour(val)) for val in p['interval']]
        p['hour_interval'] = pd.DataFrame({'hour_interval': temp})

        vital_start_temp = p['charttime'][0]
        vital_start.append(vital_start_temp)
        vital_end_temp = p['charttime'][len(p) - 1]
        vital_end.append(vital_end_temp)
        onset_time = p['onsettime'][0]
        if vital_end_temp >= onset_time:
            if vital_start_temp < onset_time:
                if timedelta_to_hour(onset_time - vital_start_temp) >= duration:
                    count_cross = count_cross + 1
                    data_x = p[vital_name].loc[
                        (p['charttime'] <= p['onsettime']) & (p['hour_interval'] <= duration)].to_numpy()
                    #print(len(data_x))
                    DT_Duration,DT_start_t,DT_end_t, OWL, DT_sign, DT_slope, DT_terminal = DominantTrendDetection(data_x, np.array(list(range(len(data_x)))))
                    #print(DT_Duration, DT_start_t,DT_end_t, OWL)
                    if ~np.isnan(DT_start_t):
                        segment_num, slope, slope_pos_percent, slope_pos_duration_percent = plot_l1_trend_fits(hadm_id,
                                                                                                               data_x,
                                                                                                               DT_start_t,
                                                                                                               DT_end_t,
                                                                                                               DT_sign,
                                                                                                               delta_values=delta_value)
'''