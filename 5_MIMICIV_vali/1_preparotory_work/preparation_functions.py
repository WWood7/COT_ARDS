import pandas as pd
import datetime
import random


def form_segments(length, test, seed):
    random.seed(seed)
    #
    # for case patients,
    # segments are just the window with specified length preceding onset
    test_case_seg = test['case'].copy()
    test_case_seg['segstart'] = pd.to_datetime(test_case_seg['onset']) + \
                                datetime.timedelta(hours=-length)
    test_case_seg['segend'] = pd.to_datetime(test_case_seg['onset'])
    test_case_seg['seg_id'] = test_case_seg['icustay_id']
    test_case_seg.reset_index(drop=True)

    #
    # for control patients,
    # for each unique icustay, if there are multiple ventilation periods recorded,
    # choose a random one
    # the segment will be the window with specified length preceding conend
    # (the end time of that ventilation period?)

    test_control_one = []
    for pat in test['control']['icustay_id'].unique():
        sub_test_control = test['control'][
            test['control']['icustay_id'] == pat].reset_index(drop=True)
        if len(sub_test_control) != 0:
            sub_test_control = sub_test_control[
                sub_test_control.index == random.randint(0, len(sub_test_control) - 1)]
            test_control_one.append(sub_test_control)
    test_control = pd.concat(test_control_one)

    test_control_seg = test_control.copy()
    test_control_seg['segstart'] = pd.to_datetime(test_control_seg['conend']) + \
                                   datetime.timedelta(hours=-length)
    test_control_seg['segend'] = pd.to_datetime(test_control_seg['conend'])
    test_control_seg['seg_id'] = test_control_seg['icustay_id']
    test_control_seg.reset_index(drop=True)

    test_segs = {'case': test_case_seg, 'control': test_control_seg}

    return test_segs