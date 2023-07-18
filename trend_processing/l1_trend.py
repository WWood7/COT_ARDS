from matplotlib import pyplot as plt
from itertools import chain
from cvxopt import matrix, solvers, spmatrix
import numpy as np
import pandas as pd
import os
from statsmodels.robust.scale import mad
import random
import scipy.stats as stats
from trend_processing.DominantTrendDetection import DominantTrendDetection, L1DominantTrendDetection

def timedelta_to_hour(time):
    d = 24*(time.days)
    h = (time.seconds)/3600
    total_hours = d + h
    return total_hours

def _first_order_derivative_matrix(size_of_matrix):
    """ Return a first order derivative matrix
    for a given signal size
    Parameters:
        size_of_matrix(int): Size of matrix
    Returns:
        first_order(cvxopt.spmatrix): Sparse matrix
        that has the first order derivative matrix
    """
    temp = size_of_matrix - 1
    first = [-1, 1] * temp
    second = list(chain.from_iterable([[ii] * 2 for ii in range(temp)]))
    third = list(chain.from_iterable([[ii, ii + 1] for ii in range(temp)]))
    first_order = spmatrix(first, second, third)

    return first_order

#print(_first_order_derivative_matrix(5))

def _lconst(signal, regularizer):
    """
    Parameters:
        signal(np.ndarray): Original, volatile signal
        regularizer(float): regularizer to keep the balance between smoothing
            and 'truthfulness' of the signal
    Returns:
        trend(np.ndarray): Trend of the signal extracted from l0 regularization
    Problem Formulation:
        minimize    (1/2) * ||x - signal||_2^2 + regularizer * sum(y)
        subject to  | D*x | <= y
    """

    signal_size = signal.size[0]
    temp = signal_size - 1
    temp_ls = range(temp)

    D = _first_order_derivative_matrix(signal_size)
    #print(D)
    P = D * D.T
    q = -D * signal

    G = spmatrix([], [], [], (2 * temp, temp))
    G[:temp, :temp] = spmatrix(1.0, temp_ls, temp_ls)
    G[temp:, :temp] = -spmatrix(1.0, temp_ls, temp_ls)
    h = matrix(regularizer, (2 * temp, 1), tc='d')
    residual = solvers.qp(P, q, G, h)
    trend = signal - D.T * residual['x']
    return trend

def lconst(signal, regularizer):
    """
    Fits the l1 trend on top of the `signal` with a particular
    `regularizer`
    Parameters:
            signal(np.ndarray): Original Signal that we want to fit l1
                trend
            regularizer(float): Regularizer which provides a balance between
                smoothing of a signal and truthfulness of signal
    Returns:
        values(np.array): L1 Trend of a signal that is extracted from the signal
    """

    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal Needs to be a numpy array")

    m = float(signal.min())
    M = float(signal.max())
    difference = M - m
    if not difference: # If signal is constant
        difference = 1
    t = (signal - m) / difference

    values = matrix(t)
    values = _lconst(values, regularizer)
    values = values * difference + m
    values = np.asarray(values).squeeze()
    return values

def plot_lconst_trend_fits(hadm_id, x, delta_values=1.):
    #plt.figure(figsize=(16, 12))
    #plt.suptitle('Different trends for different $\delta$ s')
    _FIG_DIR = 'C:/Users/duanl/OneDrive/桌面/trend/'

    #plt.subplot(len(delta_values), 1, ii + 1)
    plt.figure()
    filtered = lconst(x, delta_values)
    plt.plot(x, label='Original signal')
    label = 'Filtered, $\delta$ = {}'.format(delta_values)
    plt.plot(filtered, linewidth=5, label=label, alpha=0.5)


    plt.legend(loc='best')
    plt.title('HR of'+str(hadm_id))
    #plt.show()

    fig_name = 'hr_{}_'.format(round(hadm_id)) + '_lconst' + str(delta_values) + '.png'
    fig_path = os.path.join(_FIG_DIR, fig_name)
    plt.savefig(fig_path)
    return filtered

def _second_order_derivative_matrix(size_of_matrix):
    """ Return a second order derivative matrix
    for a given signal size
    Parameters:
        size_of_matrix(int): Size of matrix
    Returns:
        second_order(cvxopt.spmatrix): Sparse matrix
        that has the second order derivative matrix
    """
    temp = size_of_matrix - 2
    first = [1, -2, 1] * temp
    second = list(chain.from_iterable([[ii] * 3 for ii in range(temp)]))
    third = list(chain.from_iterable([[ii, ii + 1, ii + 2] for ii in range(temp)]))
    second_order = spmatrix(first, second, third)

    return second_order

#print(_second_order_derivative_matrix(5))

def _l1(signal, regularizer):
    """
    Parameters:
        signal(np.ndarray): Original, volatile signal
        regularizer(float): regularizer to keep the balance between smoothing
            and 'truthfulness' of the signal
    Returns:
        trend(np.ndarray): Trend of the signal extracted from l1 regularization
    Problem Formulation:
        minimize    (1/2) * ||x - signal||_2^2 + regularizer * sum(y)
        subject to  | D*x | <= y
    """

    signal_size = signal.size[0]
    temp = signal_size - 2
    temp_ls = range(temp)

    D = _second_order_derivative_matrix(signal_size)
    #print(D)
    P = D * D.T
    q = -D * signal

    G = spmatrix([], [], [], (2 * temp, temp))
    #print(len(G))
    G[:temp, :temp] = spmatrix(1.0, temp_ls, temp_ls)
    G[temp:, :temp] = -spmatrix(1.0, temp_ls, temp_ls)
    #print('G',G)
    h = matrix(regularizer, (2 * temp, 1), tc='d')
    #print('h', h)
    residual = solvers.qp(P, q, G, h)
    trend = signal - D.T * residual['x']
    #print(len(trend))
    kink = np.array([1 if abs(val) > 0.0001 else 0 for val in np.asarray(D * trend).squeeze()])#0.0001
    segment_num = kink.sum() + 1
    return trend, kink, segment_num

def l1(signal, regularizer):
    """
    Fits the l1 trend on top of the `signal` with a particular
    `regularizer`
    Parameters:
            signal(np.ndarray): Original Signal that we want to fit l1
                trend
            regularizer(float): Regularizer which provides a balance between
                smoothing of a signal and truthfulness of signal
    Returns:
        values(np.array): L1 Trend of a signal that is extracted from the signal
    """

    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal Needs to be a numpy array")

    m = float(signal.min())
    M = float(signal.max())
    difference = M - m
    if not difference: # If signal is constant
        difference = 1
    t = (signal - m) / difference

    values = matrix(t)
    values, kink, segment_num = _l1(values, regularizer)
    values = values * difference + m
    values = np.asarray(values).squeeze()
    kink_value = values[1:-1]*kink
    kink_value = np.insert(kink_value, 0, values[0])
    kink_value = np.append(kink_value, values[-1])
    return values, kink_value, segment_num

#def plot_l1_trend_fits(hadm_id, x, DT_start_t, DT_end_t,DT_sign,delta_values=1.):
def plot_l1_trend_fits(hadm_id, x, delta_values=1.):
    DT_Duration1,DT_start_t,DT_end_t, OWL, DT_sign1, DT_slope1, DT_terminal1 = DominantTrendDetection(x, np.array(list(range(len(x)))))
    #plt.figure(figsize=(16, 12))
    #plt.suptitle('Different trends for different $\delta$ s')
    _FIG_DIR = 'C:/Users/duanl/OneDrive/桌面/DTtrend_2/'

    #plt.subplot(len(delta_values), 1, ii + 1)
    #plt.figure()
    filtered, kink, segment_num = l1(x, delta_values)
    #plt.plot(x, label='Original signal')
    #label = 'Filtered, $\delta$ = {}'.format(delta_values)
    #plt.plot(filtered, linewidth=5, label=label, alpha=0.5)
    kink_index = list(np.nonzero(kink)[0])
    kink_value = kink[kink_index]
    #print(kink_index)
    #print(kink_value)
    #plt.plot(kink_index, kink_value, 'ro', label='Knot')
    #DT_index = [DT_start_t, DT_end_t]
    #DT_value = x[DT_index]
    #plt.plot(DT_index, DT_value, 'bo', label = 'DT'+str(DT_sign))
    slope = []
    slope_duration = []
    left_half_slope = []
    right_half_slope = []
    left_slope_duration = []
    right_slope_duration = []
    left_half_dom_dur = 0
    right_half_dom_dur = 0
    left_half_dom_terminal=np.nan
    right_half_dom_terminal=np.nan
    left_half_dom_slope=np.nan
    right_half_dom_slope=np.nan
    for i in range(len(kink_index)-1):
        temp_slope = (filtered[kink_index[i+1]] - filtered[kink_index[i]])/(kink_index[i+1] - kink_index[i])
        temp_dur = kink_index[i+1] - kink_index[i]
        if abs(temp_slope) < 0.1:
            temp_slope = 0
        slope.append(temp_slope)
        slope_duration.append(temp_dur)
        #left
        #print((kink_index[i]-kink_index[0]), len(x)/2)
        if (kink_index[i]-kink_index[0]) < len(x)/2:
            left_half_slope.append(temp_slope)
            if (kink_index[i+1]-kink_index[0]) >= len(x)/2:
                left_kink_index = kink_index[0:i+1]
                left_kink_value =np.append(list(kink[left_kink_index]), filtered[kink_index[i]] + temp_slope*((len(x)/2)-kink_index[i]))
                left_slope_duration.append(len(x)/2-kink_index[i])
                #print(left_kink_value)
                left_half_dom_dur, _, left_half_dom_slope, left_half_dom_terminal = L1DominantTrendDetection(left_half_slope, left_slope_duration, left_kink_value)
            else:
                left_slope_duration.append(temp_dur)
        #right
        if (kink_index[i+1]-kink_index[0]) > len(x)/2:
            right_half_slope.append(temp_slope)
            if (kink_index[i]-kink_index[0]) <= len(x)/2:
                right_kink_index = kink_index[i+1:]
                right_kink_value = np.insert(list(kink[right_kink_index]), 0, filtered[kink_index[i]] + temp_slope * ((len(x)/2) - kink_index[i]))
                right_slope_duration.append(kink_index[i+1] - (len(x) / 2))
                right_half_dom_dur, _, right_half_dom_slope, right_half_dom_terminal = L1DominantTrendDetection(right_half_slope, right_slope_duration, right_kink_value)
            else:
                right_slope_duration.append(temp_dur)

    #print(slope)
    pos_loc = [1 if val>0 else 0 for val in slope]
    neg_loc = [1 if val<0 else 0 for val in slope]
    slope_pos_percent = sum(pos_loc)/len(slope)
    slope_neg_percent = sum(neg_loc) / len(slope)
    slope_pos_duration = np.asarray(slope_duration) * np.asarray(pos_loc).squeeze()
    slope_neg_duration = np.asarray(slope_duration) * np.asarray(neg_loc).squeeze()
    slope_pos_duration_percent = sum(slope_pos_duration)/(len(kink)-1)
    slope_neg_duration_percent = sum(slope_neg_duration) / (len(kink) - 1)
    DT_Duration2, DT_sign2, DT_slope2, DT_terminal2 = L1DominantTrendDetection(slope, slope_duration, kink_value)
    #plt.legend(loc='best')
    #plt.title('HR of'+str(hadm_id))
    #plt.show()

    #fig_name = 'hr_{}_'.format(round(hadm_id)) + '_DT' + str(delta_values) + '.png'
    #fig_path = os.path.join(_FIG_DIR, fig_name)
    #plt.savefig(fig_path)
    return segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent,\
           kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, right_half_dom_dur, \
           left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, DT_Duration2, \
           DT_terminal2, DT_slope2
'''
def _l0_l1(signal, regularizer_l0, regularizer_l1):
    signal_len = signal.size[0]
    D1 = _first_order_derivative_matrix(signal_len)
    D2 = _second_order_derivative_matrix(signal_len)
    beta = cp.Variable(signal_len)
    obj = cp.Minimize((1/2)*cp.sum_squares(signal-beta)+regularizer_l0*cp.norm(D1*beta, p=1)+regularizer_l1*cp.norm(D2*beta, p=1))
    prob = cp.Problem(obj)
    prob.solve()
    trend = beta.value
    kink = np.array([1 if abs(val) > 0.0001 else 0 for val in np.asarray(D2 * trend).squeeze()])  # 0.0001
    segment_num = kink.sum() + 1
    return trend, kink, segment_num

def l0_l1(signal, regularizer_l0, regularizer_l1):
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal Needs to be a numpy array")

    m = float(signal.min())
    M = float(signal.max())
    difference = M - m
    if not difference: # If signal is constant
        difference = 1
    t = (signal - m) / difference

    values, kink, segment_num = _l0_l1(t, regularizer_l0, regularizer_l1)
    values = values * difference + m
    values = np.asarray(values).squeeze()
    kink_value = values[1:-1]*kink
    kink_value = np.insert(kink_value, 0, values[0])
    kink_value = np.append(kink_value, values[-1])
    return values, kink_value, segment_num

def plot_l0_l1_trend_fits(hadm_id, x, delta_values1=1., delta_values2=1.):
    DT_Duration1,DT_start_t,DT_end_t, OWL, DT_sign1, DT_slope1, DT_terminal1 = DominantTrendDetection(x, np.array(list(range(len(x)))))
    #plt.figure(figsize=(16, 12))
    #plt.suptitle('Different trends for different $\delta$ s')
    _FIG_DIR = 'C:/Users/duanl/OneDrive/桌面/DTtrend_2/'

    #plt.subplot(len(delta_values), 1, ii + 1)
    #plt.figure()
    filtered, kink, segment_num = l0_l1(x, delta_values1, delta_values2)
    #plt.plot(x, label='Original signal')
    #label = 'Filtered, $\delta$ = {}'.format(delta_values)
    #plt.plot(filtered, linewidth=5, label=label, alpha=0.5)
    kink_index = list(np.nonzero(kink)[0])
    kink_value = kink[kink_index]
    #print(kink_index)
    #print(kink_value)
    #plt.plot(kink_index, kink_value, 'ro', label='Knot')
    #DT_index = [DT_start_t, DT_end_t]
    #DT_value = x[DT_index]
    #plt.plot(DT_index, DT_value, 'bo', label = 'DT'+str(DT_sign))
    slope = []
    slope_duration = []
    left_half_slope = []
    right_half_slope = []
    left_slope_duration = []
    right_slope_duration = []
    left_half_dom_dur = 0
    right_half_dom_dur = 0
    left_half_dom_terminal=np.nan
    right_half_dom_terminal=np.nan
    left_half_dom_slope=np.nan
    right_half_dom_slope=np.nan
    for i in range(len(kink_index)-1):
        temp_slope = (filtered[kink_index[i+1]] - filtered[kink_index[i]])/(kink_index[i+1] - kink_index[i])
        temp_dur = kink_index[i+1] - kink_index[i]
        if abs(temp_slope) < 0.1:
            temp_slope = 0
        slope.append(temp_slope)
        slope_duration.append(temp_dur)
        #left
        #print((kink_index[i]-kink_index[0]), len(x)/2)
        if (kink_index[i]-kink_index[0]) < len(x)/2:
            left_half_slope.append(temp_slope)
            if (kink_index[i+1]-kink_index[0]) >= len(x)/2:
                left_kink_index = kink_index[0:i+1]
                left_kink_value =np.append(list(kink[left_kink_index]), filtered[kink_index[i]] + temp_slope*((len(x)/2)-kink_index[i]))
                left_slope_duration.append(len(x)/2-kink_index[i])
                #print(left_kink_value)
                left_half_dom_dur, _, left_half_dom_slope, left_half_dom_terminal = L1DominantTrendDetection(left_half_slope, left_slope_duration, left_kink_value)
            else:
                left_slope_duration.append(temp_dur)
        #right
        if (kink_index[i+1]-kink_index[0]) > len(x)/2:
            right_half_slope.append(temp_slope)
            if (kink_index[i]-kink_index[0]) <= len(x)/2:
                right_kink_index = kink_index[i+1:]
                right_kink_value = np.insert(list(kink[right_kink_index]), 0, filtered[kink_index[i]] + temp_slope * ((len(x)/2) - kink_index[i]))
                right_slope_duration.append(kink_index[i+1] - (len(x) / 2))
                right_half_dom_dur, _, right_half_dom_slope, right_half_dom_terminal = L1DominantTrendDetection(right_half_slope, right_slope_duration, right_kink_value)
            else:
                right_slope_duration.append(temp_dur)

    #print(slope)
    pos_loc = [1 if val>0 else 0 for val in slope]
    neg_loc = [1 if val<0 else 0 for val in slope]
    slope_pos_percent = sum(pos_loc)/len(slope)
    slope_neg_percent = sum(neg_loc) / len(slope)
    slope_pos_duration = np.asarray(slope_duration) * np.asarray(pos_loc).squeeze()
    slope_neg_duration = np.asarray(slope_duration) * np.asarray(neg_loc).squeeze()
    slope_pos_duration_percent = sum(slope_pos_duration)/(len(kink)-1)
    slope_neg_duration_percent = sum(slope_neg_duration) / (len(kink) - 1)
    DT_Duration2, DT_sign2, DT_slope2, DT_terminal2 = L1DominantTrendDetection(slope, slope_duration, kink_value)
    #plt.legend(loc='best')
    #plt.title('HR of'+str(hadm_id))
    #plt.show()

    #fig_name = 'hr_{}_'.format(round(hadm_id)) + '_DT' + str(delta_values) + '.png'
    #fig_path = os.path.join(_FIG_DIR, fig_name)
    #plt.savefig(fig_path)
    return segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent,\
           kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, right_half_dom_dur, \
           left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, DT_Duration2, \
           DT_terminal2, DT_slope2
'''

def _third_order_derivative_matrix(size_of_matrix):
    """ Return a first order derivative matrix
    for a given signal size
    Parameters:
        size_of_matrix(int): Size of matrix
    Returns:
        first_order(cvxopt.spmatrix): Sparse matrix
        that has the first order derivative matrix
    """
    temp = size_of_matrix - 3
    first = [-1, 3, -3, 1] * temp
    second = list(chain.from_iterable([[ii] * 4 for ii in range(temp)]))
    third = list(chain.from_iterable([[ii, ii + 1, ii + 2, ii + 3] for ii in range(temp)]))
    first_order = spmatrix(first, second, third)

    return first_order

#print(_first_order_derivative_matrix(5))

def _l2(signal, regularizer):
    """
    Parameters:
        signal(np.ndarray): Original, volatile signal
        regularizer(float): regularizer to keep the balance between smoothing
            and 'truthfulness' of the signal
    Returns:
        trend(np.ndarray): Trend of the signal extracted from l0 regularization
    Problem Formulation:
        minimize    (1/2) * ||x - signal||_2^2 + regularizer * sum(y)
        subject to  | D*x | <= y
    """

    signal_size = signal.size[0]
    temp = signal_size - 3
    temp_ls = range(temp)

    D = _third_order_derivative_matrix(signal_size)
    #print(D)
    P = D * D.T
    q = -D * signal

    G = spmatrix([], [], [], (2 * temp, temp))
    G[:temp, :temp] = spmatrix(1.0, temp_ls, temp_ls)
    G[temp:, :temp] = -spmatrix(1.0, temp_ls, temp_ls)
    h = matrix(regularizer, (2 * temp, 1), tc='d')
    residual = solvers.qp(P, q, G, h)
    trend = signal - D.T * residual['x']
    return trend

def l2(signal, regularizer):
    """
    Fits the l1 trend on top of the `signal` with a particular
    `regularizer`
    Parameters:
            signal(np.ndarray): Original Signal that we want to fit l1
                trend
            regularizer(float): Regularizer which provides a balance between
                smoothing of a signal and truthfulness of signal
    Returns:
        values(np.array): L1 Trend of a signal that is extracted from the signal
    """

    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal Needs to be a numpy array")

    m = float(signal.min())
    M = float(signal.max())
    difference = M - m
    if not difference: # If signal is constant
        difference = 1
    t = (signal - m) / difference

    values = matrix(t)
    values = _l2(values, regularizer)
    values = values * difference + m
    values = np.asarray(values).squeeze()
    return values

def plot_l2_trend_fits(hadm_id, x, delta_values=1.):
    #plt.figure(figsize=(16, 12))
    #plt.suptitle('Different trends for different $\delta$ s')
    _FIG_DIR = 'C:/Users/duanl/OneDrive/桌面/trend/'

    #plt.subplot(len(delta_values), 1, ii + 1)
    plt.figure()
    filtered = l2(x, delta_values)
    plt.plot(x, label='Original signal')
    label = 'Filtered, $\delta$ = {}'.format(delta_values)
    plt.plot(filtered, linewidth=5, label=label, alpha=0.5)


    plt.legend(loc='best')
    plt.title('HR of'+str(hadm_id))
    #plt.show()

    fig_name = 'hr_{}_'.format(round(hadm_id)) + '_l2' + str(delta_values) + '.png'
    fig_path = os.path.join(_FIG_DIR, fig_name)
    plt.savefig(fig_path)
    return filtered

def strip_outliers(original_signal, delta, mad_coef=3):
    """
    Based on l1 trend filtering, this function provides an endpoint
    """
    filtered_t = l1(original_signal, delta)

    diff = original_signal - filtered_t.squeeze()
    median_of_difference = np.median(diff)
    mad_of_difference = mad(diff)
    filtered_signal = original_signal.copy()
    threshold = mad_coef * mad_of_difference
    filtered_signal[np.abs(diff - median_of_difference) > threshold] = np.nan
    #filtered_signal = pd.Series(filtered_signal).fillna(method='ffill').fillna(method='bfill')

    return filtered_signal

def plot_outlier_removal_via_l1(outlier_signal, mad_coefficients=None):
    plt.figure(figsize=(16, 12))
    plt.suptitle('Outlier detection via l1 with different MAD coefficient')
    if mad_coefficients is None:
        mad_coefficients = range(1, 4)

    for ii, mad_coef in enumerate(mad_coefficients):
        plt.subplot(len(mad_coefficients), 1, ii + 1)
        x_wo_outliers = strip_outliers(outlier_signal, delta=1, mad_coef=mad_coef)
        plt.plot(outlier_signal, label='Original signal')
        label = 'Stripped Outliers, mad_coef = {}'.format(mad_coef)
        plt.plot(x_wo_outliers, linewidth=5, label=label, alpha=0.5)
        plt.legend(loc='best')
######################################################################ards_get_trend_feature
def ards_get_trend_feature(imputed_vital_ards, train_ards_hadm_onset, vital_name, duration, delta_value, method):
    train_ards_hadm_onset = train_ards_hadm_onset.rename(columns={"charttime": "onsettime"})
    # add onset time to vitals
    vital_ards = pd.merge(
        imputed_vital_ards[['hadm_id', 'charttime', vital_name]], train_ards_hadm_onset[['hadm_id', 'onsettime']],
        how="inner", on=["hadm_id"])
    print(len(train_ards_hadm_onset), len(vital_ards['hadm_id'].unique()), "should be the same")
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

    # now we have 40 features
    feature_table = pd.DataFrame(columns=['hadm_id',  'segment_num',
                                          'slope_pos_max', 'slope_pos_min', 'slope_pos_median', 'slope_pos_mean',
                                          'slope_neg_max', 'slope_neg_min', 'slope_neg_median', 'slope_neg_mean',
                                          'slope_pos_percent', 'slope_pos_duration_percent',
                                          'slope_neg_percent', 'slope_neg_duration_percent',
                                          'pos_slope_max_min_ratio', 'neg_slope_max_min_ratio',
                                          'slope_change_rate_gt10_num', 'slope_change_rate_gt20_num',
                                          'slope_change_rate_gt30_num', 'slope_change_rate_gt40_num',
                                          'slope_change_rate_gt50_num', 'slope_change_rate_gt60_num',
                                          'slope_change_rate_gt70_num', 'slope_change_rate_gt80_num',
                                          'slope_change_rate_gt90_num', 'slope_change_rate_gt100_num',
                                          'terminal_max', 'terminal_min', 'terminal_median', 'terminal_mean',
                                          'DTposdur1', 'DTnegdur1', 'DTterminal1', 'DTslope1',
                                          'DTposdur2', 'DTnegdur2', 'DTterminal2', 'DTslope2',
                                          'th_DTterminal_ratio','th_DTslope_lastup_ratio', 'th_DTslope_lastdown_ratio'])

    # six situation counts
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
        vital_end_temp = p['charttime'][len(p)-1]
        onset_time = p['onsettime'][0]
        if vital_end_temp < onset_time:
            if timedelta_to_hour(onset_time - vital_end_temp) >= duration:
                count_total_before = count_total_before + 1
            else:
                if timedelta_to_hour(onset_time - vital_start_temp) > duration:
                    count_half_before =  count_half_before + 1
                else:
                    count_middle = count_middle + 1
            feature_table = feature_table.append(pd.DataFrame({'hadm_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
        else:
            if vital_start_temp >= onset_time:
                count_total_after = count_total_after + 1
                feature_table = feature_table.append(pd.DataFrame({'hadm_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
            else:
                if timedelta_to_hour(onset_time - vital_start_temp) >= duration:
                    count_cross = count_cross + 1
                    data_x = p[vital_name].loc[(p['charttime'] <= p['onsettime']) & (p['hour_interval'] <= duration)].to_numpy()
                    print(len(data_x))
                    if method=="l1":
                        segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent, \
                        kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, \
                        right_half_dom_dur, left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, \
                        DT_Duration2, DT_terminal2, DT_slope2 = plot_l1_trend_fits(hadm_id,data_x,delta_values=delta_value)
                    else:
                        segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent, \
                        kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, \
                        right_half_dom_dur, left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, \
                        DT_Duration2, DT_terminal2, DT_slope2 = plot_l1_trend_fits(hadm_id, data_x,
                                                                                   delta_values=delta_value)

                    neg_slope = [abs(val) for val in slope if val < 0]
                    pos_slope = [val for val in slope if val > 0]
                    slope_change_rate = []
                    for slope_idx in range(len(slope)-1):
                        if slope[slope_idx]==0:
                            if slope[slope_idx+1]!=0:
                                slope_change_rate.append(1)#this is beacuse 'slope_change_rate_gt100_num' is the maximal %
                            else:
                                slope_change_rate.append(0)
                        else:
                            slope_change_rate.append(abs(slope[slope_idx+1]-slope[slope_idx]/slope[slope_idx]))

                    if right_half_dom_slope != np.nan:
                        if left_half_dom_slope==np.nan:
                            th_DTslope_lastup_ratio = 400
                            th_DTslope_lastdown_ratio = 400
                        else:
                            if right_half_dom_slope > 0:
                                th_DTslope_lastup_ratio = right_half_dom_slope/left_half_dom_slope
                                th_DTslope_lastdown_ratio = 0
                            else:
                                th_DTslope_lastdown_ratio = right_half_dom_slope/left_half_dom_slope
                                th_DTslope_lastup_ratio = 0
                    else:
                        th_DTslope_lastup_ratio = 0
                        th_DTslope_lastdown_ratio = 0


                    #print(data_x)

                    #if len(neg_slope) != 0:
                    #    print(neg_slope)
                    #    print(np.max(neg_slope), np.min(neg_slope))
                    #if len(pos_slope) != 0:
                    #    print(pos_slope)
                    #    print(np.max(pos_slope), np.min(pos_slope))

                    feature_table = feature_table.append(pd.DataFrame({'hadm_id': [hadm_id], 'segment_num': [segment_num],
                                                                       'slope_pos_max': [np.max(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_min': [np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_median': [np.median(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_mean': [np.mean(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_neg_max': [np.max(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_min': [np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_median': [np.median(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_mean': [np.mean(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_pos_percent': [slope_pos_percent],
                                                                       'slope_pos_duration_percent': [slope_pos_duration_percent],
                                                                       'slope_neg_percent': [slope_neg_percent],
                                                                       'slope_neg_duration_percent': [slope_neg_duration_percent],
                                                                       'pos_slope_max_min_ratio': [np.max(pos_slope)/np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'neg_slope_max_min_ratio': [np.max(neg_slope) / np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_change_rate_gt10_num': [sum([1 if val >=0.1 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt20_num': [sum([1 if val >=0.2 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt30_num': [sum([1 if val >=0.3 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt40_num': [sum([1 if val >=0.4 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt50_num': [sum([1 if val >=0.5 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt60_num': [sum([1 if val >=0.6 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt70_num': [sum([1 if val >=0.7 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt80_num': [sum([1 if val >=0.8 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt90_num': [sum([1 if val >=0.9 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt100_num': [sum([1 if val >=1 else 0 for val in slope_change_rate])],
                                                                       'terminal_max': [np.max(kink_value)], 'terminal_min': [np.min(kink_value)],
                                                                       'terminal_median': [np.median(kink_value)],'terminal_mean': [np.mean(kink_value)],
                                                                       'DTposdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 > 0 else 0],
                                                                       'DTnegdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 < 0 else 0],
                                                                       'DTterminal1': [DT_terminal1],
                                                                       'DTslope1': [DT_slope1],
                                                                       'DTposdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 > 0 else 0],
                                                                       'DTnegdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 < 0 else 0],
                                                                       'DTterminal2': [DT_terminal2],
                                                                       'DTslope2': [DT_slope2],
                                                                       'th_DTterminal_ratio': [right_half_dom_terminal/left_half_dom_terminal],
                                                                       'th_DTslope_lastup_ratio': [th_DTslope_lastup_ratio],
                                                                       'th_DTslope_lastdown_ratio': [th_DTslope_lastdown_ratio]
                                                                       }),
                                                         ignore_index=True)
                else:
                    count_half_after = count_half_after + 1
                    feature_table = feature_table.append(pd.DataFrame({'hadm_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
    print(count_total_before, count_half_before, count_middle, count_half_after, count_total_after, count_cross)
    #plt.hist(vital_lengh, bins=6)
    #plt.show()

    print(len(feature_table))
    return feature_table

def case_get_trend_feature(imputed_vital_ards, train_ards_hadm_onset, vital_name, duration, delta_value, method):
    # add onset time to vitals
    vital_ards = pd.merge(
        imputed_vital_ards[['icustay_id', 'charttime', vital_name]], train_ards_hadm_onset[['icustay_id', 'onset']],
        how="inner", on=["icustay_id"])
    print(len(train_ards_hadm_onset), len(vital_ards['icustay_id'].unique()), "should be the same")
    print(vital_ards.head())
    print(vital_ards[vital_name].mean(), vital_ards[vital_name].median(), vital_ards[vital_name].max(),
          vital_ards[vital_name].min())

    uniperIDs = vital_ards['icustay_id'].unique()
    uniperIDs.sort()
    dataHadmID = []
    for id in range(len(uniperIDs)):
        dataHadmID.append(vital_ards[vital_ards['icustay_id'] == uniperIDs[id]])
    for i in range(len(dataHadmID)):
        dataHadmID[i] = dataHadmID[i].sort_values(by=['charttime'])
    print(len(dataHadmID), 'encounters')

    # now we have 40 features
    feature_table = pd.DataFrame(columns=['icustay_id',  'segment_num',
                                          'slope_pos_max', 'slope_pos_min', 'slope_pos_median', 'slope_pos_mean',
                                          'slope_neg_max', 'slope_neg_min', 'slope_neg_median', 'slope_neg_mean',
                                          'slope_pos_percent', 'slope_pos_duration_percent',
                                          'slope_neg_percent', 'slope_neg_duration_percent',
                                          'pos_slope_max_min_ratio', 'neg_slope_max_min_ratio',
                                          'slope_change_rate_gt10_num', 'slope_change_rate_gt20_num',
                                          'slope_change_rate_gt30_num', 'slope_change_rate_gt40_num',
                                          'slope_change_rate_gt50_num', 'slope_change_rate_gt60_num',
                                          'slope_change_rate_gt70_num', 'slope_change_rate_gt80_num',
                                          'slope_change_rate_gt90_num', 'slope_change_rate_gt100_num',
                                          'terminal_max', 'terminal_min', 'terminal_median', 'terminal_mean',
                                          'DTposdur1', 'DTnegdur1', 'DTterminal1', 'DTslope1',
                                          'DTposdur2', 'DTnegdur2', 'DTterminal2', 'DTslope2',
                                          'th_DTterminal_ratio','th_DTslope_lastup_ratio', 'th_DTslope_lastdown_ratio'])

    # six situation counts
    count_total_before = 0
    count_half_before = 0
    count_middle = 0
    count_half_after = 0
    count_total_after = 0
    count_cross = 0
    for p in dataHadmID:
        p = p.reset_index(drop=True)
        hadm_id = p['icustay_id'][0]
        p['charttime'] = pd.to_datetime(p['charttime'])
        p['onset'] = pd.to_datetime(p['onset'])
        p['interval'] = p['onset'] - p['charttime']
        temp = [abs(timedelta_to_hour(val)) for val in p['interval']]
        p['hour_interval'] = pd.DataFrame({'hour_interval': temp})

        vital_start_temp = p['charttime'][0]
        vital_end_temp = p['charttime'][len(p)-1]
        onset_time = p['onset'][0]
        if vital_end_temp < onset_time:
            if timedelta_to_hour(onset_time - vital_end_temp) >= duration:
                count_total_before = count_total_before + 1
            else:
                if timedelta_to_hour(onset_time - vital_start_temp) > duration:
                    count_half_before =  count_half_before + 1
                else:
                    count_middle = count_middle + 1
            feature_table = feature_table.append(pd.DataFrame({'icustay_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
        else:
            if vital_start_temp >= onset_time:
                count_total_after = count_total_after + 1
                feature_table = feature_table.append(pd.DataFrame({'icustay_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
            else:
                if timedelta_to_hour(onset_time - vital_start_temp) >= duration:
                    count_cross = count_cross + 1
                    data_x = p[vital_name].loc[(p['charttime'] <= p['onset']) & (p['hour_interval'] <= duration)].to_numpy()
                    print(len(data_x))
                    if method=="l1":
                        segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent, \
                        kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, \
                        right_half_dom_dur, left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, \
                        DT_Duration2, DT_terminal2, DT_slope2 = plot_l1_trend_fits(hadm_id,data_x,delta_values=delta_value)
                    else:
                        segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent, \
                        kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, \
                        right_half_dom_dur, left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, \
                        DT_Duration2, DT_terminal2, DT_slope2 = plot_l1_trend_fits(hadm_id, data_x,
                                                                                   delta_values=delta_value)

                    neg_slope = [abs(val) for val in slope if val < 0]
                    pos_slope = [val for val in slope if val > 0]
                    slope_change_rate = []
                    for slope_idx in range(len(slope)-1):
                        if slope[slope_idx]==0:
                            if slope[slope_idx+1]!=0:
                                slope_change_rate.append(1)#this is beacuse 'slope_change_rate_gt100_num' is the maximal %
                            else:
                                slope_change_rate.append(0)
                        else:
                            slope_change_rate.append(abs(slope[slope_idx+1]-slope[slope_idx]/slope[slope_idx]))

                    if right_half_dom_slope != np.nan:
                        if left_half_dom_slope==np.nan:
                            th_DTslope_lastup_ratio = 400
                            th_DTslope_lastdown_ratio = 400
                        else:
                            if right_half_dom_slope > 0:
                                th_DTslope_lastup_ratio = right_half_dom_slope/left_half_dom_slope
                                th_DTslope_lastdown_ratio = 0
                            else:
                                th_DTslope_lastdown_ratio = right_half_dom_slope/left_half_dom_slope
                                th_DTslope_lastup_ratio = 0
                    else:
                        th_DTslope_lastup_ratio = 0
                        th_DTslope_lastdown_ratio = 0


                    #print(data_x)

                    #if len(neg_slope) != 0:
                    #    print(neg_slope)
                    #    print(np.max(neg_slope), np.min(neg_slope))
                    #if len(pos_slope) != 0:
                    #    print(pos_slope)
                    #    print(np.max(pos_slope), np.min(pos_slope))

                    feature_table = feature_table.append(pd.DataFrame({'icustay_id': [hadm_id], 'segment_num': [segment_num],
                                                                       'slope_pos_max': [np.max(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_min': [np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_median': [np.median(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_mean': [np.mean(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_neg_max': [np.max(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_min': [np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_median': [np.median(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_mean': [np.mean(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_pos_percent': [slope_pos_percent],
                                                                       'slope_pos_duration_percent': [slope_pos_duration_percent],
                                                                       'slope_neg_percent': [slope_neg_percent],
                                                                       'slope_neg_duration_percent': [slope_neg_duration_percent],
                                                                       'pos_slope_max_min_ratio': [np.max(pos_slope)/np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'neg_slope_max_min_ratio': [np.max(neg_slope) / np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_change_rate_gt10_num': [sum([1 if val >=0.1 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt20_num': [sum([1 if val >=0.2 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt30_num': [sum([1 if val >=0.3 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt40_num': [sum([1 if val >=0.4 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt50_num': [sum([1 if val >=0.5 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt60_num': [sum([1 if val >=0.6 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt70_num': [sum([1 if val >=0.7 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt80_num': [sum([1 if val >=0.8 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt90_num': [sum([1 if val >=0.9 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt100_num': [sum([1 if val >=1 else 0 for val in slope_change_rate])],
                                                                       'terminal_max': [np.max(kink_value)], 'terminal_min': [np.min(kink_value)],
                                                                       'terminal_median': [np.median(kink_value)],'terminal_mean': [np.mean(kink_value)],
                                                                       'DTposdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 > 0 else 0],
                                                                       'DTnegdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 < 0 else 0],
                                                                       'DTterminal1': [DT_terminal1],
                                                                       'DTslope1': [DT_slope1],
                                                                       'DTposdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 > 0 else 0],
                                                                       'DTnegdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 < 0 else 0],
                                                                       'DTterminal2': [DT_terminal2],
                                                                       'DTslope2': [DT_slope2],
                                                                       'th_DTterminal_ratio': [right_half_dom_terminal/left_half_dom_terminal],
                                                                       'th_DTslope_lastup_ratio': [th_DTslope_lastup_ratio],
                                                                       'th_DTslope_lastdown_ratio': [th_DTslope_lastdown_ratio]
                                                                       }),
                                                         ignore_index=True)
                else:
                    count_half_after = count_half_after + 1
                    feature_table = feature_table.append(pd.DataFrame({'icustay_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
    print(count_total_before, count_half_before, count_middle, count_half_after, count_total_after, count_cross)
    #plt.hist(vital_lengh, bins=6)
    #plt.show()

    print(len(feature_table))
    return feature_table
######################################################################nonards_get_trend_feature
def nonards_get_trend_feature(imputed_vital_nonards, train_nonards_seg_hadm_start_end, vital_name, duration, delta_value):
    print(len(imputed_vital_nonards['hadm_id'].unique()), len(train_nonards_seg_hadm_start_end['seg_id'].unique()))
    vital_nonards = pd.merge(
        imputed_vital_nonards[['hadm_id', 'charttime', vital_name]], train_nonards_seg_hadm_start_end[['seg_id', 'hadm_id', 'start', 'end']],
        how="inner", on=["hadm_id"]).drop_duplicates(keep='first')
    #vital_nonards = vital_nonards[
    #    (vital_nonards['charttime'] <= vital_nonards['end']) & (vital_nonards['charttime'] >= vital_nonards['start'])]
    uniperIDs = vital_nonards['seg_id'].unique()
    uniperIDs.sort()
    datasegID = []
    for id in range(len(uniperIDs)):
        datasegID.append(vital_nonards[vital_nonards['seg_id'] == uniperIDs[id]])
    for i in range(len(datasegID)):
        datasegID[i] = datasegID[i].sort_values(by=['charttime'])
    print(len(datasegID), 'segemnts')
    print(vital_nonards[vital_name].mean(), vital_nonards[vital_name].median(),
          vital_nonards[vital_name].max(), vital_nonards[vital_name].min())

    slope_dic = {}
    last_slope_list = []
    count_total_before = 0
    count_half_before = 0
    count_middle = 0
    count_half_after = 0
    count_total_after = 0
    count_cross = 0
    feature_table = pd.DataFrame(columns=['seg_id', 'icustay_id', 'segment_num',
                                          'slope_pos_max', 'slope_pos_min', 'slope_pos_median', 'slope_pos_mean',
                                          'slope_neg_max', 'slope_neg_min', 'slope_neg_median', 'slope_neg_mean',
                                          'slope_pos_percent', 'slope_pos_duration_percent',
                                          'slope_neg_percent', 'slope_neg_duration_percent',
                                          'pos_slope_max_min_ratio', 'neg_slope_max_min_ratio',
                                          'slope_change_rate_gt10_num', 'slope_change_rate_gt20_num',
                                          'slope_change_rate_gt30_num', 'slope_change_rate_gt40_num',
                                          'slope_change_rate_gt50_num', 'slope_change_rate_gt60_num',
                                          'slope_change_rate_gt70_num', 'slope_change_rate_gt80_num',
                                          'slope_change_rate_gt90_num', 'slope_change_rate_gt100_num',
                                          'terminal_max', 'terminal_min', 'terminal_median', 'terminal_mean',
                                          'DTposdur1', 'DTnegdur1', 'DTterminal1', 'DTslope1',
                                          'DTposdur2', 'DTnegdur2', 'DTterminal2', 'DTslope2',
                                          'th_DTterminal_ratio','th_DTslope_lastup_ratio', 'th_DTslope_lastdown_ratio'])
    for p in datasegID:
        p = p.reset_index(drop=True)
        seg_id = p['seg_id'][0]
        hadm_id = p['hadm_id'][0]
        p['charttime'] = pd.to_datetime(p['charttime'])
        p['end'] = pd.to_datetime(p['end'])
        p['interval'] = p['end'] - p['charttime']
        temp = [timedelta_to_hour(val) for val in p['interval']]
        p['hour_interval'] = pd.DataFrame({'hour_interval': temp})

        vital_start_temp = p['charttime'][0]
        vital_end_temp = p['charttime'][len(p) - 1]
        end_time = p['end'][0]
        if vital_end_temp < end_time:
            if timedelta_to_hour(end_time - vital_end_temp) >= duration:
                count_total_before = count_total_before + 1
            else:
                if timedelta_to_hour(end_time - vital_start_temp) > duration:
                    count_half_before = count_half_before + 1
                else:
                    count_middle = count_middle + 1
            feature_table = feature_table.append(pd.DataFrame({'seg_id': [seg_id], 'hadm_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
        else:
            if vital_start_temp >= end_time:
                count_total_after = count_total_after + 1
                feature_table = feature_table.append(pd.DataFrame({'seg_id': [seg_id], 'hadm_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
            else:
                if timedelta_to_hour(end_time - vital_start_temp) >= duration:
                    count_cross = count_cross + 1
                    data_x = p[vital_name].loc[
                        (p['charttime'] <= p['end']) & (p['hour_interval'] <= duration)].to_numpy()
                    print(len(data_x))
                    segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent, \
                    kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, \
                    right_half_dom_dur, left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, \
                    DT_Duration2, DT_terminal2, DT_slope2 = plot_l1_trend_fits(seg_id,data_x,delta_values=delta_value)

                    neg_slope = [abs(val) for val in slope if val < 0]
                    pos_slope = [val for val in slope if val > 0]
                    slope_change_rate = []
                    for slope_idx in range(len(slope) - 1):
                        if slope[slope_idx] == 0:
                            if slope[slope_idx + 1] != 0:
                                slope_change_rate.append(
                                    1)  # this is beacuse 'slope_change_rate_gt100_num' is the maximal %
                            else:
                                slope_change_rate.append(0)
                        else:
                            slope_change_rate.append(abs(slope[slope_idx + 1] - slope[slope_idx] / slope[slope_idx]))

                    if right_half_dom_slope != np.nan:
                        if left_half_dom_slope == np.nan:
                            th_DTslope_lastup_ratio = 400
                            th_DTslope_lastdown_ratio = 400
                        else:
                            if right_half_dom_slope > 0:
                                th_DTslope_lastup_ratio = right_half_dom_slope / left_half_dom_slope
                                th_DTslope_lastdown_ratio = 0
                            else:
                                th_DTslope_lastdown_ratio = right_half_dom_slope / left_half_dom_slope
                                th_DTslope_lastup_ratio = 0
                    else:
                        th_DTslope_lastup_ratio = 0
                        th_DTslope_lastdown_ratio = 0
                    # print(data_x)

                    # if len(neg_slope) != 0:
                    #    print(neg_slope)
                    #    print(np.max(neg_slope), np.min(neg_slope))
                    # if len(pos_slope) != 0:
                    #    print(pos_slope)
                    #    print(np.max(pos_slope), np.min(pos_slope))

                    feature_table = feature_table.append(
                        pd.DataFrame({'seg_id': [seg_id], 'hadm_id': [hadm_id], 'segment_num': [segment_num],
                                                                       'slope_pos_max': [np.max(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_min': [np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_median': [np.median(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_mean': [np.mean(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_neg_max': [np.max(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_min': [np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_median': [np.median(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_mean': [np.mean(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_pos_percent': [slope_pos_percent],
                                                                       'slope_pos_duration_percent': [slope_pos_duration_percent],
                                                                       'slope_neg_percent': [slope_neg_percent],
                                                                       'slope_neg_duration_percent': [slope_neg_duration_percent],
                                                                       'pos_slope_max_min_ratio': [np.max(pos_slope)/np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'neg_slope_max_min_ratio': [np.max(neg_slope) / np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_change_rate_gt10_num': [sum([1 if val >=0.1 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt20_num': [sum([1 if val >=0.2 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt30_num': [sum([1 if val >=0.3 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt40_num': [sum([1 if val >=0.4 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt50_num': [sum([1 if val >=0.5 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt60_num': [sum([1 if val >=0.6 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt70_num': [sum([1 if val >=0.7 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt80_num': [sum([1 if val >=0.8 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt90_num': [sum([1 if val >=0.9 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt100_num': [sum([1 if val >=1 else 0 for val in slope_change_rate])],
                                                                       'terminal_max': [np.max(kink_value)], 'terminal_min': [np.min(kink_value)],
                                                                       'terminal_median': [np.median(kink_value)],'terminal_mean': [np.mean(kink_value)],
                                                                       'DTposdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 > 0 else 0],
                                                                       'DTnegdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 < 0 else 0],
                                                                       'DTterminal1': [DT_terminal1],
                                                                       'DTslope1': [DT_slope1],
                                                                       'DTposdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 > 0 else 0],
                                                                       'DTnegdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 < 0 else 0],
                                                                       'DTterminal2': [DT_terminal2],
                                                                       'DTslope2': [DT_slope2],
                                                                       'th_DTterminal_ratio': [right_half_dom_terminal/left_half_dom_terminal],
                                                                       'th_DTslope_lastup_ratio': [th_DTslope_lastup_ratio],
                                                                       'th_DTslope_lastdown_ratio': [th_DTslope_lastdown_ratio]}),
                        ignore_index=True)
                else:
                    count_half_after = count_half_after + 1
                    feature_table = feature_table.append(pd.DataFrame({'seg_id': [seg_id], 'hadm_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
    print(count_total_before, count_half_before, count_middle, count_half_after, count_total_after, count_cross)
    print(len(feature_table))
    return feature_table

def control_get_trend_feature(imputed_vital_nonards, train_nonards_seg_hadm_start_end, vital_name, duration, delta_value):
    print(len(imputed_vital_nonards['icustay_id'].unique()), len(train_nonards_seg_hadm_start_end['seg_id'].unique()))
    vital_nonards = pd.merge(
        imputed_vital_nonards[['icustay_id', 'charttime', vital_name]], train_nonards_seg_hadm_start_end[['seg_id', 'icustay_id', 'segstart', 'segend']],
        how="inner", on=["icustay_id"]).drop_duplicates(keep='first')
    #vital_nonards = vital_nonards[
    #    (vital_nonards['charttime'] <= vital_nonards['end']) & (vital_nonards['charttime'] >= vital_nonards['start'])]
    uniperIDs = vital_nonards['seg_id'].unique()
    uniperIDs.sort()
    datasegID = []
    for id in range(len(uniperIDs)):
        datasegID.append(vital_nonards[vital_nonards['seg_id'] == uniperIDs[id]])
    for i in range(len(datasegID)):
        datasegID[i] = datasegID[i].sort_values(by=['charttime'])
    print(len(datasegID), 'segemnts')
    print(vital_nonards[vital_name].mean(), vital_nonards[vital_name].median(),
          vital_nonards[vital_name].max(), vital_nonards[vital_name].min())

    slope_dic = {}
    last_slope_list = []
    count_total_before = 0
    count_half_before = 0
    count_middle = 0
    count_half_after = 0
    count_total_after = 0
    count_cross = 0
    feature_table = pd.DataFrame(columns=['seg_id', 'icustay_id', 'segment_num',
                                          'slope_pos_max', 'slope_pos_min', 'slope_pos_median', 'slope_pos_mean',
                                          'slope_neg_max', 'slope_neg_min', 'slope_neg_median', 'slope_neg_mean',
                                          'slope_pos_percent', 'slope_pos_duration_percent',
                                          'slope_neg_percent', 'slope_neg_duration_percent',
                                          'pos_slope_max_min_ratio', 'neg_slope_max_min_ratio',
                                          'slope_change_rate_gt10_num', 'slope_change_rate_gt20_num',
                                          'slope_change_rate_gt30_num', 'slope_change_rate_gt40_num',
                                          'slope_change_rate_gt50_num', 'slope_change_rate_gt60_num',
                                          'slope_change_rate_gt70_num', 'slope_change_rate_gt80_num',
                                          'slope_change_rate_gt90_num', 'slope_change_rate_gt100_num',
                                          'terminal_max', 'terminal_min', 'terminal_median', 'terminal_mean',
                                          'DTposdur1', 'DTnegdur1', 'DTterminal1', 'DTslope1',
                                          'DTposdur2', 'DTnegdur2', 'DTterminal2', 'DTslope2',
                                          'th_DTterminal_ratio','th_DTslope_lastup_ratio', 'th_DTslope_lastdown_ratio'])
    for p in datasegID:
        p = p.reset_index(drop=True)
        seg_id = p['seg_id'][0]
        hadm_id = p['icustay_id'][0]
        p['charttime'] = pd.to_datetime(p['charttime'])
        p['segend'] = pd.to_datetime(p['segend'])
        p['interval'] = p['segend'] - p['charttime']
        temp = [timedelta_to_hour(val) for val in p['interval']]
        p['hour_interval'] = pd.DataFrame({'hour_interval': temp})

        vital_start_temp = p['charttime'][0]
        vital_end_temp = p['charttime'][len(p) - 1]
        end_time = p['segend'][0]
        if vital_end_temp < end_time:
            if timedelta_to_hour(end_time - vital_end_temp) >= duration:
                count_total_before = count_total_before + 1
            else:
                if timedelta_to_hour(end_time - vital_start_temp) > duration:
                    count_half_before = count_half_before + 1
                else:
                    count_middle = count_middle + 1
            feature_table = feature_table.append(pd.DataFrame({'seg_id': [seg_id], 'icustay_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
        else:
            if vital_start_temp >= end_time:
                count_total_after = count_total_after + 1
                feature_table = feature_table.append(pd.DataFrame({'seg_id': [seg_id], 'icustay_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
            else:
                if timedelta_to_hour(end_time - vital_start_temp) >= duration:
                    count_cross = count_cross + 1
                    data_x = p[vital_name].loc[
                        (p['charttime'] <= p['segend']) & (p['hour_interval'] <= duration)].to_numpy()
                    print(len(data_x))
                    segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent, \
                    kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, \
                    right_half_dom_dur, left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, \
                    DT_Duration2, DT_terminal2, DT_slope2 = plot_l1_trend_fits(seg_id,data_x,delta_values=delta_value)

                    neg_slope = [abs(val) for val in slope if val < 0]
                    pos_slope = [val for val in slope if val > 0]
                    slope_change_rate = []
                    for slope_idx in range(len(slope) - 1):
                        if slope[slope_idx] == 0:
                            if slope[slope_idx + 1] != 0:
                                slope_change_rate.append(
                                    1)  # this is beacuse 'slope_change_rate_gt100_num' is the maximal %
                            else:
                                slope_change_rate.append(0)
                        else:
                            slope_change_rate.append(abs(slope[slope_idx + 1] - slope[slope_idx] / slope[slope_idx]))

                    if right_half_dom_slope != np.nan:
                        if left_half_dom_slope == np.nan:
                            th_DTslope_lastup_ratio = 400
                            th_DTslope_lastdown_ratio = 400
                        else:
                            if right_half_dom_slope > 0:
                                th_DTslope_lastup_ratio = right_half_dom_slope / left_half_dom_slope
                                th_DTslope_lastdown_ratio = 0
                            else:
                                th_DTslope_lastdown_ratio = right_half_dom_slope / left_half_dom_slope
                                th_DTslope_lastup_ratio = 0
                    else:
                        th_DTslope_lastup_ratio = 0
                        th_DTslope_lastdown_ratio = 0
                    # print(data_x)

                    # if len(neg_slope) != 0:
                    #    print(neg_slope)
                    #    print(np.max(neg_slope), np.min(neg_slope))
                    # if len(pos_slope) != 0:
                    #    print(pos_slope)
                    #    print(np.max(pos_slope), np.min(pos_slope))

                    feature_table = feature_table.append(
                        pd.DataFrame({'seg_id': [seg_id], 'icustay_id': [hadm_id], 'segment_num': [segment_num],
                                                                       'slope_pos_max': [np.max(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_min': [np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_median': [np.median(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_pos_mean': [np.mean(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'slope_neg_max': [np.max(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_min': [np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_median': [np.median(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_neg_mean': [np.mean(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_pos_percent': [slope_pos_percent],
                                                                       'slope_pos_duration_percent': [slope_pos_duration_percent],
                                                                       'slope_neg_percent': [slope_neg_percent],
                                                                       'slope_neg_duration_percent': [slope_neg_duration_percent],
                                                                       'pos_slope_max_min_ratio': [np.max(pos_slope)/np.min(pos_slope) if len(pos_slope) != 0 else 0],
                                                                       'neg_slope_max_min_ratio': [np.max(neg_slope) / np.min(neg_slope) if len(neg_slope) != 0 else 0],
                                                                       'slope_change_rate_gt10_num': [sum([1 if val >=0.1 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt20_num': [sum([1 if val >=0.2 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt30_num': [sum([1 if val >=0.3 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt40_num': [sum([1 if val >=0.4 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt50_num': [sum([1 if val >=0.5 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt60_num': [sum([1 if val >=0.6 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt70_num': [sum([1 if val >=0.7 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt80_num': [sum([1 if val >=0.8 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt90_num': [sum([1 if val >=0.9 else 0 for val in slope_change_rate])],
                                                                       'slope_change_rate_gt100_num': [sum([1 if val >=1 else 0 for val in slope_change_rate])],
                                                                       'terminal_max': [np.max(kink_value)], 'terminal_min': [np.min(kink_value)],
                                                                       'terminal_median': [np.median(kink_value)],'terminal_mean': [np.mean(kink_value)],
                                                                       'DTposdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 > 0 else 0],
                                                                       'DTnegdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 < 0 else 0],
                                                                       'DTterminal1': [DT_terminal1],
                                                                       'DTslope1': [DT_slope1],
                                                                       'DTposdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 > 0 else 0],
                                                                       'DTnegdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 < 0 else 0],
                                                                       'DTterminal2': [DT_terminal2],
                                                                       'DTslope2': [DT_slope2],
                                                                       'th_DTterminal_ratio': [right_half_dom_terminal/left_half_dom_terminal],
                                                                       'th_DTslope_lastup_ratio': [th_DTslope_lastup_ratio],
                                                                       'th_DTslope_lastdown_ratio': [th_DTslope_lastdown_ratio]}),
                        ignore_index=True)
                else:
                    count_half_after = count_half_after + 1
                    feature_table = feature_table.append(pd.DataFrame({'seg_id': [seg_id], 'icustay_id': [hadm_id], 'segment_num': [-1],
                                          'slope_pos_max': [-1], 'slope_pos_min': [-1], 'slope_pos_median': [-1], 'slope_pos_mean': [-1],
                                          'slope_neg_max': [-1], 'slope_neg_min': [-1], 'slope_neg_median': [-1], 'slope_neg_mean': [-1],
                                          'slope_pos_percent': [-1], 'slope_pos_duration_percent': [-1],
                                          'slope_neg_percent': [-1], 'slope_neg_duration_percent': [-1],
                                          'pos_slope_max_min_ratio': [-1], 'neg_slope_max_min_ratio': [-1],
                                          'slope_change_rate_gt10_num': [-1], 'slope_change_rate_gt20_num': [-1],
                                          'slope_change_rate_gt30_num': [-1], 'slope_change_rate_gt40_num': [-1],
                                          'slope_change_rate_gt50_num': [-1], 'slope_change_rate_gt60_num': [-1],
                                          'slope_change_rate_gt70_num': [-1], 'slope_change_rate_gt80_num': [-1],
                                          'slope_change_rate_gt90_num': [-1], 'slope_change_rate_gt100_num': [-1],
                                          'terminal_max': [-1], 'terminal_min': [-1], 'terminal_median': [-1], 'terminal_mean': [-1],
                                          'DTposdur1': [-1], 'DTnegdur1': [-1], 'DTterminal1': [-1], 'DTslope1': [-1],
                                          'DTposdur2': [-1], 'DTnegdur2': [-1], 'DTterminal2': [-1], 'DTslope2': [-1],
                                          'th_DTterminal_ratio': [-1],'th_DTslope_lastup_ratio': [-1], 'th_DTslope_lastdown_ratio': [-1]}),
                                          ignore_index=True)
    print(count_total_before, count_half_before, count_middle, count_half_after, count_total_after, count_cross)
    print(len(feature_table))
    return feature_table
###############################################get mean and var
def get_mean_var(vital_ards_trend_dic, vital_nonards_trend_dic):
    k = 2
    n = len(vital_ards_trend_dic)
    d1 = k - 1
    d2 = k * (n - 1)

    # nonards
    non_all_slope_list = []
    non_all_kink_num = []
    non_all_slope_pos_perc = []
    non_all_slope_pos_duration_perc = []
    for key in vital_nonards_trend_dic:
        non_all_slope_list = non_all_slope_list + [np.mean(vital_nonards_trend_dic[key][:-3])]
        non_all_kink_num = non_all_kink_num + [vital_nonards_trend_dic[key][-3]]
        non_all_slope_pos_perc = non_all_slope_pos_perc + [vital_nonards_trend_dic[key][-2]]
        non_all_slope_pos_duration_perc = non_all_slope_pos_duration_perc + [vital_nonards_trend_dic[key][-1]]

    random.seed(2021)
    non_all_slope_list = random.sample(non_all_slope_list, n)
    non_all_kink_num = random.sample(non_all_kink_num, n)
    non_all_slope_pos_perc = random.sample(non_all_slope_pos_perc, n)
    non_all_slope_pos_duration_perc = random.sample(non_all_slope_pos_duration_perc, n)

    non_slope_mean = np.mean(non_all_slope_list)
    non_slope_var = np.var(non_all_slope_list)
    non_kink_num_mean = np.mean(non_all_kink_num)
    non_kink_num_var = np.var(non_all_kink_num)
    non_slope_pos_perc_mean = np.mean(non_all_slope_pos_perc)
    non_slope_pos_perc_var = np.var(non_all_slope_pos_perc)
    non_slope_pos_duration_perc_mean = np.mean(non_all_slope_pos_duration_perc)
    non_slope_pos_duration_perc_var = np.var(non_all_slope_pos_duration_perc)

    # ards
    all_slope_list = []
    all_kink_num = []
    all_slope_pos_perc = []
    all_slope_pos_duration_perc = []
    for key in vital_ards_trend_dic:
        all_slope_list = all_slope_list + [np.mean(vital_ards_trend_dic[key][:-3])]
        all_kink_num = all_kink_num + [vital_ards_trend_dic[key][-3]]
        all_slope_pos_perc = all_slope_pos_perc + [vital_ards_trend_dic[key][-2]]
        all_slope_pos_duration_perc = all_slope_pos_duration_perc + [vital_ards_trend_dic[key][-1]]

    slope_mean = np.mean(all_slope_list)
    slope_var = np.var(all_slope_list)
    kink_num_mean = np.mean(all_kink_num)
    kink_num_var = np.var(all_kink_num)
    slope_pos_perc_mean = np.mean(all_slope_pos_perc)
    slope_pos_perc_var = np.var(all_slope_pos_perc)
    slope_pos_duration_perc_mean = np.mean(all_slope_pos_duration_perc)
    slope_pos_duration_perc_var = np.var(all_slope_pos_duration_perc)

    temp_slope_mean = (non_slope_mean + slope_mean)/k
    msb_slope = n * ((pow((non_slope_mean-temp_slope_mean),2)+pow((slope_mean-temp_slope_mean),2))/(k-1))
    temp_kink_num_mean = (non_kink_num_mean + kink_num_mean)/k
    msb_kink_num = n * ((pow((non_kink_num_mean-temp_kink_num_mean),2)+pow((kink_num_mean-temp_kink_num_mean),2))/(k-1))
    temp_slope_pos_perc_mean = (non_slope_pos_perc_mean + slope_pos_perc_mean)/k
    msb_slope_pos_perc = n * ((pow((non_slope_pos_perc_mean-temp_slope_pos_perc_mean),2)+pow((slope_pos_perc_mean-temp_slope_pos_perc_mean),2))/(k-1))
    temp_slope_pos_duration_perc_mean = (non_slope_pos_duration_perc_mean + slope_pos_duration_perc_mean)/k
    msb_slope_pos_duration_perc =n * ((pow((non_slope_pos_duration_perc_mean-temp_slope_pos_duration_perc_mean),2)+pow((slope_pos_duration_perc_mean-temp_slope_pos_duration_perc_mean),2))/(k-1))

    mse_slope = (non_slope_var + slope_var)/k
    mse_kink_num = (non_kink_num_var + kink_num_var)/k
    mse_slope_pos_perc = (non_slope_pos_perc_var + slope_pos_perc_var)/k
    mse_slope_pos_duration_perc = (non_slope_pos_duration_perc_var + slope_pos_duration_perc_var)/k

    f_slope = msb_slope/mse_slope
    f_kink_num = msb_kink_num/mse_kink_num
    f_slope_pos_perc = msb_slope_pos_perc/mse_slope_pos_perc
    f_slope_pos_duration_perc = msb_slope_pos_duration_perc/mse_slope_pos_duration_perc

    print(d1, d2)
    print(non_slope_mean, non_slope_var, non_kink_num_mean, non_kink_num_var, non_slope_pos_perc_mean,
          non_slope_pos_perc_var, non_slope_pos_duration_perc_mean, non_slope_pos_duration_perc_var)
    print(slope_mean, slope_var, kink_num_mean, kink_num_var, slope_pos_perc_mean, slope_pos_perc_var,
          slope_pos_duration_perc_mean, slope_pos_duration_perc_var)

    return d1, d2, f_slope, f_kink_num, f_slope_pos_perc, f_slope_pos_duration_perc

###############################################get Welch's t-test
def get_welch_t_test(vital_ards_trend_dic, vital_nonards_trend_dic):
    # nonards
    non_all_slope_list_mean = []
    non_all_slope_list_median = []
    non_all_slope_list_max = []
    non_all_slope_list_min = []
    non_all_kink_num = []
    non_all_slope_pos_perc = []
    non_all_slope_pos_duration_perc = []
    for key in vital_nonards_trend_dic:
        non_all_slope_list_mean = non_all_slope_list_mean + [np.mean(vital_nonards_trend_dic[key][:-3])]
        non_all_slope_list_median = non_all_slope_list_median + [np.median(vital_nonards_trend_dic[key][:-3])]
        non_all_slope_list_max = non_all_slope_list_max + [np.max(vital_nonards_trend_dic[key][:-3])]
        non_all_slope_list_min = non_all_slope_list_min + [np.min(vital_nonards_trend_dic[key][:-3])]
        non_all_kink_num = non_all_kink_num + [vital_nonards_trend_dic[key][-3]]
        non_all_slope_pos_perc = non_all_slope_pos_perc + [vital_nonards_trend_dic[key][-2]]
        non_all_slope_pos_duration_perc = non_all_slope_pos_duration_perc + [vital_nonards_trend_dic[key][-1]]

    # ards
    all_slope_list_mean = []
    all_slope_list_median = []
    all_slope_list_max = []
    all_slope_list_min = []
    all_kink_num = []
    all_slope_pos_perc = []
    all_slope_pos_duration_perc = []
    for key in vital_ards_trend_dic:
        all_slope_list_mean = all_slope_list_mean + [np.mean(vital_ards_trend_dic[key][:-3])]
        all_slope_list_median = all_slope_list_median + [np.median(vital_ards_trend_dic[key][:-3])]
        all_slope_list_max = all_slope_list_max + [np.max(vital_ards_trend_dic[key][:-3])]
        all_slope_list_min = all_slope_list_min + [np.min(vital_ards_trend_dic[key][:-3])]
        all_kink_num = all_kink_num + [vital_ards_trend_dic[key][-3]]
        all_slope_pos_perc = all_slope_pos_perc + [vital_ards_trend_dic[key][-2]]
        all_slope_pos_duration_perc = all_slope_pos_duration_perc + [vital_ards_trend_dic[key][-1]]

    slope_mean_mean = np.mean(non_all_slope_list_mean + all_slope_list_mean)
    slope_mean_std = np.std(non_all_slope_list_mean + all_slope_list_mean)
    non_slope_mean_norm = (non_all_slope_list_mean - slope_mean_mean)/slope_mean_std
    slope_mean_norm = (all_slope_list_mean - slope_mean_mean) / slope_mean_std

    slope_median_mean = np.mean(non_all_slope_list_median + all_slope_list_median)
    slope_median_std = np.std(non_all_slope_list_median + all_slope_list_median)
    non_slope_median_norm = (non_all_slope_list_median - slope_median_mean) / slope_median_std
    slope_median_norm = (all_slope_list_median - slope_median_mean) / slope_median_std

    slope_max_mean = np.mean(non_all_slope_list_max + all_slope_list_max)
    slope_max_std = np.std(non_all_slope_list_max + all_slope_list_max)
    non_slope_max_norm = (non_all_slope_list_max - slope_max_mean) / slope_max_std
    slope_max_norm = (all_slope_list_max - slope_max_mean) / slope_max_std

    slope_min_mean = np.mean(non_all_slope_list_min + all_slope_list_min)
    slope_min_std = np.std(non_all_slope_list_min + all_slope_list_min)
    non_slope_min_norm = (non_all_slope_list_min - slope_min_mean) / slope_min_std
    slope_min_norm = (all_slope_list_min - slope_min_mean) / slope_min_std

    kink_num_mean = np.mean(non_all_kink_num + all_kink_num)
    kink_num_std = np.std(non_all_kink_num + all_kink_num)
    non_kink_num_norm = (non_all_kink_num - kink_num_mean) / kink_num_std
    kink_num_norm = (all_kink_num - kink_num_mean) / kink_num_std

    slope_pos_perc_mean = np.mean(non_all_slope_pos_perc + all_slope_pos_perc)
    slope_pos_perc_std = np.std(non_all_slope_pos_perc + all_slope_pos_perc)
    non_all_slope_pos_perc_norm = (non_all_slope_pos_perc - slope_pos_perc_mean) / slope_pos_perc_std
    all_slope_pos_perc_norm = (all_slope_pos_perc - slope_pos_perc_mean) / slope_pos_perc_std

    slope_pos_duration_perc_mean = np.mean(non_all_slope_pos_duration_perc + all_slope_pos_duration_perc)
    slope_pos_duration_perc_std = np.std(non_all_slope_pos_duration_perc + all_slope_pos_duration_perc)
    non_all_slope_pos_duration_perc_norm = (non_all_slope_pos_duration_perc - slope_pos_duration_perc_mean) / slope_pos_duration_perc_std
    all_slope_pos_duration_perc_norm = (all_slope_pos_duration_perc - slope_pos_duration_perc_mean) / slope_pos_duration_perc_std

    _, slope_mean_p = stats.ttest_ind(non_slope_mean_norm, slope_mean_norm, equal_var=False)
    _, slope_median_p = stats.ttest_ind(non_slope_median_norm, slope_median_norm, equal_var=False)
    _, slope_max_p = stats.ttest_ind(non_slope_max_norm, slope_max_norm, equal_var=False)
    _, slope_min_p = stats.ttest_ind(non_slope_min_norm, slope_min_norm, equal_var=False)
    _, kink_num_p = stats.ttest_ind(non_kink_num_norm, kink_num_norm, equal_var=False)
    _, slope_pos_perc_p = stats.ttest_ind(non_all_slope_pos_perc_norm, all_slope_pos_perc_norm, equal_var=False)
    _, slope_pos_duration_perc_p = stats.ttest_ind(non_all_slope_pos_duration_perc_norm, all_slope_pos_duration_perc_norm, equal_var=False)
    print(slope_mean_p, slope_median_p, slope_max_p, slope_min_p, kink_num_p, slope_pos_perc_p, slope_pos_duration_perc_p)
    return slope_mean_p, slope_median_p, slope_max_p, slope_min_p, kink_num_p, slope_pos_perc_p, slope_pos_duration_perc_p