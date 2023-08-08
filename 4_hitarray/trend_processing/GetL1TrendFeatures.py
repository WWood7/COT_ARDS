from itertools import chain
from cvxopt import matrix, solvers, spmatrix
import numpy as np
from trend_processing.DominantTrendDetection import DominantTrendDetection, L1DominantTrendDetection

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

    filtered, kink, segment_num = l1(x, delta_values)

    kink_index = list(np.nonzero(kink)[0])
    kink_value = kink[kink_index]
    #print(kink_index)
    #print(kink_value)

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

    return segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, slope_neg_duration_percent,\
           kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, right_half_slope, left_half_dom_dur, right_half_dom_dur, \
           left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, right_half_dom_slope, DT_Duration2, \
           DT_terminal2, DT_slope2